"""JSON-files contract fix, step 3 (GPU): a small supervised fine-tune.

Each step: a few JSON-contract sequences (loss only on the rewritten visible answer; thinking is context) and a few
replay sequences (the model's own finished answers in other formats, loss = negative log-probability under the TEST
sampler, which is zero-gradient in expectation at the start and so acts as an anchor). fp32 master, AdamW as in
grpo.py, linear warm-up, bf16 export by stochastic rounding at the listed steps.

usage: sft_json.py INIT_DIR OUT_DIR --data TRAIN.jsonl [--steps 150 --lr 2e-6 --save-at 50,100,150]
       sft_json.py INIT_DIR - --data TRAIN.jsonl --check     (tokenisation and masking only, no model, CPU)
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("LOGIT_CHUNK", "256")
import torch

import grpo

PRESENCE = 1.5


def encode(tok, row, max_len):
    """prompt ids, completion ids, index of the first supervised completion token; None if too long"""
    rendered = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    p = tok(rendered, add_special_tokens=False).input_ids
    if row["kind"] == "json":
        think, tail = row["prefix"][: row["prefix"].rfind("</think>") + len("</think>")], row["prefix"][row["prefix"].rfind("</think>") + len("</think>"):]
        a = tok(think, add_special_tokens=False).input_ids
        b = tok(tail + row["target"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        c, start = a + b, len(a)
    else:
        c, start = tok(row["target"], add_special_tokens=False).input_ids + [tok.eos_token_id], 0
    if len(p) + len(c) > max_len:
        return None
    return p, c, start


def seq_loss(model, embed, ids, n_prompt, start, matched):
    """mean negative log-probability of completion tokens from `start` on. matched=True: probability under the test sampler
    (presence penalty, top-k, top-p; tokens outside the kept set carry no gradient). matched=False: presence-adjusted
    softmax over the whole vocabulary (an upper bound on the sampler's loss that also pulls a token into the kept set)."""
    hidden = model.model.language_model(input_ids=ids).last_hidden_state[0, n_prompt - 1:-1]
    targets = ids[0, n_prompt:]
    n = targets.shape[0]
    first_seen = torch.full((embed.shape[0],), n, dtype=torch.long, device=targets.device)
    first_seen.scatter_reduce_(0, targets, torch.arange(n, device=targets.device), reduce="amin")
    total, count = 0.0, 0
    for s in range(start, n, grpo.CHUNK):
        def chunk(h, t, pos):
            logits = (h @ embed.T).float()
            if matched:
                lp, valid = grpo.sampler_logprob(logits, t, pos, first_seen)
                return (lp * valid).sum(), valid.sum().float()
            adj = logits - PRESENCE * (first_seen[None, :] < pos[:, None]).to(logits.dtype)
            return adj.log_softmax(-1).gather(-1, t[:, None])[:, 0].sum(), torch.tensor(float(t.numel()), device=t.device)
        pos = torch.arange(s, min(s + grpo.CHUNK, n), device=targets.device)
        lp, c = torch.utils.checkpoint.checkpoint(chunk, hidden[s:s + grpo.CHUNK], targets[s:s + grpo.CHUNK], pos, use_reentrant=False)
        total = total + lp
        count = count + c
    return -total / count.clamp_min(1.0), int(count.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("init"); p.add_argument("out"); p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=150); p.add_argument("--lr", type=float, default=2e-6); p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repaired", type=int, default=5); p.add_argument("--valid", type=int, default=5); p.add_argument("--replay", type=int, default=6)
    p.add_argument("--save-at", default="50,100,150"); p.add_argument("--max-len", type=int, default=6144); p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.init)
    rows = [json.loads(l) for l in open(a.data)]
    groups = {"repaired": [], "valid": [], "replay": []}
    skipped = 0
    for r in rows:
        e = encode(tok, r, a.max_len)
        if e is None:
            skipped += 1
            continue
        groups["replay" if r["kind"] == "replay" else ("valid" if r["how"] == "valid" else "repaired")].append((e, r))
    print({k: len(v) for k, v in groups.items()}, "too long, skipped:", skipped, flush=True)
    if a.check:
        for g in ("repaired", "valid", "replay"):
            (pi, c, start), r = groups[g][0]
            print(f"--- {g}: prompt {len(pi)} tokens, completion {len(c)}, supervised from {start} ({len(c) - start} tokens)")
            print("    last context tokens:", repr(tok.decode(c[max(0, start - 6):start])), "| first supervised:", repr(tok.decode(c[start:start + 24])), "| last:", repr(tok.decode(c[-6:])))
        import statistics as st
        for g, v in groups.items():
            print(g, "median supervised tokens", int(st.median(len(e[1]) - e[2] for e, _ in v)), "median total", int(st.median(len(e[0]) + len(e[1]) for e, _ in v)))
        return
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(Path(a.init) / "model.safetensors"), "pt") as f:
        dtypes = {k: {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[f.get_slice(k).get_dtype()] for k in f.keys()}
    model = AutoModelForImageTextToText.from_pretrained(a.init, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    embed = model.model.language_model.embed_tokens.weight
    held = set(model.state_dict())
    with safe_open(str(Path(a.init) / "model.safetensors"), "pt") as f:
        frozen = {k: f.get_tensor(k) for k in dtypes if k not in held}
    sr_gen = torch.Generator(device="cuda").manual_seed(a.seed)
    rng = random.Random(a.seed)
    order = {g: [] for g in groups}
    def draw(g, k):
        got = []
        while len(got) < k:
            if not order[g]:
                order[g] = list(range(len(groups[g]))); rng.shuffle(order[g])
            got.append(groups[g][order[g].pop()][0])
        return got
    save_at = {int(s) for s in a.save_at.split(",") if s}
    log = open(out / "train_log.jsonl", "a")
    for step in range(1, a.steps + 1):
        t0 = time.time()
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1.0, step / max(1, a.warmup))
        batch = [("repaired", e) for e in draw("repaired", a.repaired)] + [("valid", e) for e in draw("valid", a.valid)] + [("replay", e) for e in draw("replay", a.replay)]
        opt.zero_grad(set_to_none=True)
        sums = {"repaired": [0.0, 0], "valid": [0.0, 0], "replay": [0.0, 0]}
        for g, (pi, c, start) in batch:
            ids = torch.tensor([pi + c], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, n = seq_loss(model, embed, ids, len(pi), start, matched=(g == "replay"))
            if n:
                (loss / len(batch)).backward()
                sums[g][0] += float(loss); sums[g][1] += 1
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        stats = {"step": step, **{f"loss_{g}": round(v[0] / max(1, v[1]), 4) for g, v in sums.items()}, "grad_norm": round(gnorm, 4),
                 "lr": opt.param_groups[0]["lr"], "step_s": round(time.time() - t0, 1)}
        print(json.dumps(stats), flush=True); log.write(json.dumps(stats) + "\n"); log.flush()
        if step in save_at:
            ck = out / f"step{step}"; ck.mkdir(exist_ok=True)
            for f in Path(a.init).iterdir():
                if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"):
                    shutil.copy(f, ck / f.name)
            save_file(grpo.export_state(model, frozen, dtypes, sr_gen), str(ck / "model.safetensors"), metadata={"format": "pt"})
            print(json.dumps({"saved": str(ck)}), flush=True)


if __name__ == "__main__":
    main()
