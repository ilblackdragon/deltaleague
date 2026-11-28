"""Continue the round-one Delta LoRA using the round-two completion-only trainer."""
import argparse, hashlib, json, math, os, random, re, time
import torch, torch.utils.checkpoint
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"]


def chunk_nll(head, h, t):
    return torch.nn.functional.cross_entropy(head(h).float(), t, reduction="sum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--submission", required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=7168); ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--initial-adapter", required=True)
    ap.add_argument("--checkpoints", type=int, nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.parent)
    ex = [json.loads(l) for f in a.data for l in open(f)]
    random.shuffle(ex)
    if a.limit:
        ex = ex[: a.limit]
    data, dropped = [], 0
    for e in ex:
        p = tok(e["prompt"], add_special_tokens=False).input_ids
        c = tok(e["completion"] + "<|im_end|>", add_special_tokens=False).input_ids
        if len(p) + len(c) > a.max_len:
            dropped += 1; continue
        data.append((p, c, e.get("kind", "?")))
    kinds = {k: sum(1 for d in data if d[2] == k) for k in {d[2] for d in data}}
    print(f"examples {len(data)} (dropped {dropped} over {a.max_len} tokens); by kind {kinds}; "
          f"completion tokens {sum(len(d[1]) for d in data)}", flush=True)

    dt = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.parent, dtype=dt).to(a.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none",
                     exclude_modules=r".*(visual|mtp).*")
    model = get_peft_model(model, cfg)
    initialize_adapter(model, a)
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    steps = math.ceil(len(data) / a.batch)
    if max(a.checkpoints) != steps or min(a.checkpoints) < 1 or dropped:
        raise ValueError(f"Pilot schedule expected checkpoints through step {steps}, with zero dropped rows")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    backbone = model.base_model.model.model
    head = model.base_model.model.lm_head
    model.train()
    t0 = time.time()
    for s in range(steps):
        batch = data[s * a.batch:(s + 1) * a.batch]
        ntok = sum(len(c) for _, c, _ in batch)
        tot = 0.0
        for p, c, _ in batch:
            ids = torch.tensor([p + c], device=a.device)
            h = backbone(input_ids=ids).last_hidden_state[0, len(p) - 1:-1]
            tgt = ids[0, len(p):]
            loss = 0.0
            for k in range(0, h.shape[0], 1024):
                loss = loss + torch.utils.checkpoint.checkpoint(chunk_nll, head, h[k:k + 1024], tgt[k:k + 1024], use_reentrant=False)
            (loss / ntok).backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        line = {"step": s, "of": steps, "loss_per_token": tot / ntok, "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
        print(json.dumps(line), flush=True)
        with open(f"{a.out}/train_log.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        if s + 1 in a.checkpoints:
            export_adapter(model, a, f"{a.submission}/step-{s + 1}")


def initialize_adapter(model, args):
    from pathlib import Path

    path = Path(args.initial_adapter)
    manifest = json.loads((path / "manifest.json").read_text())
    parent = json.loads((Path(args.parent) / "delta-model.json").read_text())
    if (manifest["format"] != "lora" or manifest["parent_version_id"] != parent["id"]
            or manifest["parent_content_hash"] != parent["content_hash"]):
        raise ValueError("Warm-start adapter has a different parent")
    artifact, = manifest["artifacts"]
    weights = path / artifact["name"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Warm-start adapter checksum mismatch")
    targets = manifest["tensors"]
    loaded = set()
    with safe_open(weights, "pt") as source, torch.no_grad():
        for name, mod in model.named_modules():
            if not hasattr(mod, "lora_A") or "default" not in getattr(mod, "lora_A", {}):
                continue
            match = re.search(r"layers\.(\d+)\.(.+)$", name)
            key = f"model.language_model.layers.{match.group(1)}.{match.group(2)}.weight"
            spec = targets[key]
            if spec["alpha"] != args.alpha:
                raise ValueError(f"Warm-start scale differs for {key}")
            for label, target in (("a", mod.lora_A["default"].weight), ("b", mod.lora_B["default"].weight)):
                initial = source.get_tensor(spec[label])
                if target.shape != initial.shape or target.dtype != initial.dtype:
                    raise ValueError(f"Warm-start shape or dtype differs for {key}/{label}")
                target.copy_(initial)
            loaded.add(key)
    if loaded != targets.keys():
        raise ValueError("Warm-start adapter target coverage differs")
    print(f"Warm-started all {len(loaded)} round-one LoRA targets", flush=True)


def export_adapter(model, a, destination):
    # ---- export Delta `lora` patch
    with safe_open(f"{a.parent}/model.safetensors", "pt") as f:
        ckeys = set(f.keys()); shapes = {k: f.get_slice(k).get_shape() for k in ckeys}
    tensors, spec = {}, {}
    for name, mod in model.named_modules():
        if not hasattr(mod, "lora_A") or "default" not in getattr(mod, "lora_A", {}):
            continue
        m = re.search(r"layers\.(\d+)\.(.+)$", name)
        key = f"model.language_model.layers.{m.group(1)}.{m.group(2)}.weight"
        assert key in ckeys, key
        A = mod.lora_A["default"].weight.detach().float().cpu().contiguous()
        B = mod.lora_B["default"].weight.detach().float().cpu().contiguous()
        assert (B.shape[0], A.shape[1]) == tuple(shapes[key]), (key, A.shape, B.shape, shapes[key])
        i = len(spec)
        tensors[f"t{i}.a"], tensors[f"t{i}.b"] = A, B
        spec[key] = {"a": f"t{i}.a", "b": f"t{i}.b", "alpha": float(a.alpha)}
    os.makedirs(destination, exist_ok=False)
    wpath = f"{destination}/weights.safetensors"
    save_file(tensors, wpath)
    meta = json.load(open(f"{a.parent}/delta-model.json"))
    manifest = {"schema_version": "delta.submission.v1", "reconstruction_version": "delta-reconstruct-v1",
                "model_config_id": meta["model_id"], "parent_version_id": meta["id"],
                "parent_content_hash": meta["content_hash"], "format": "lora", "operation": "replace",
                "artifacts": [{"name": "weights.safetensors", "sha256": hashlib.sha256(open(wpath, "rb").read()).hexdigest(),
                               "size_bytes": os.path.getsize(wpath)}],
                "tensors": dict(sorted(spec.items()))}
    json.dump(manifest, open(f"{destination}/manifest.json", "w"), indent=2)
    print(f"exported {len(spec)} LoRA targets to {destination} ({os.path.getsize(wpath)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
