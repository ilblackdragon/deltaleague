"""Gate-only GRPO for Qwen3.5-0.8B (Delta League qwen35-08b-v1).

Only the linear-attention memory/erase gates are trainable:
  layers.*.linear_attn.{A_log, dt_bias, in_proj_a.weight, in_proj_b.weight}   (18 layers, ~0.6M params)
Everything else is frozen. Rollouts come from a vLLM server (podman, ROCm) at the private-test sampler
(thinking on, T=1.0, top_p 0.95, top_k 20, presence 1.5), with a shorter practice cap. After each update the
gate tensors are written into a copy of the checkpoint and vLLM is restarted on it (on-policy, 1 update/batch).

Reward: 1 + 0.2 * (1 - tokens/cap) if the answer is correct AND thinking closed AND generation stopped
naturally (else 0), plus 0.25 for closing thinking and stopping naturally even if wrong (densifies the
signal: the parent closes thinking within 8K on only 7% of held-out practice problems). Group-normalised advantages (G samples per prompt); zero-variance groups are skipped.
Loss: -A * sum(token logprob) / cap (constant normaliser, no length bias). Master weights in fp32, served copy
is round-to-nearest bf16 of the master (the exported tensor is exactly what was sampled from).

Usage (host, ROCm torch): python grpo_gates.py --parent model-orig --data data --out runs/gates --steps 15
Practice data: TRAINING splits only (prep_data.py). Never uses public validation or private data.
"""
import argparse, json, math, os, random, shutil, subprocess, time, re, concurrent.futures as cf, urllib.request
import torch, torch.utils.checkpoint
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

GATE_RE = re.compile(r"layers\.(\d+)\.linear_attn\.(A_log|dt_bias|in_proj_a\.weight|in_proj_b\.weight)$")
IMAGE = "docker.io/kyuz0/vllm-therock-gfx1151:latest"
PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def log(run, msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(f"{run}/log.txt", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- vLLM server
def start_vllm(model_dir, port, util, max_seqs=128):
    subprocess.run(["podman", "rm", "-f", "dl-vllm"], capture_output=True)
    cmd = ["podman", "run", "-d", "--rm", "--name", "dl-vllm", "--device", "/dev/kfd", "--device", "/dev/dri",
           "--group-add", "keep-groups", "--security-opt", "seccomp=unconfined", "--ipc=host",
           "-p", f"127.0.0.1:{port}:8000", "-v", f"{os.path.abspath(model_dir)}:/model:ro",
           "-v", f"{os.path.expanduser('~')}/.cache/vllm-container:/cache",
           "-e", "TRITON_CACHE_DIR=/cache/triton", "-e", "VLLM_CACHE_ROOT=/cache/vllm",
           "-e", "VLLM_ROCM_USE_AITER=0", "-e", "VLLM_ROCM_USE_AITER_LINEAR=0", IMAGE,
           "vllm", "serve", "/model", "--served-model-name", "policy", "--host", "0.0.0.0", "--port", "8000",
           "--language-model-only", "--dtype", "bfloat16", "--max-model-len", "16384", "--max-num-seqs", str(max_seqs),
           "--gpu-memory-utilization", str(util), "--attention-backend", "TRITON_ATTN", "--no-enable-log-requests"]
    subprocess.run(cmd, check=True, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < 1200:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return time.time() - t0
        except Exception:
            st = subprocess.run(["podman", "inspect", "-f", "{{.State.Running}}", "dl-vllm"], capture_output=True, text=True)
            if st.stdout.strip() != "true":
                raise RuntimeError("vLLM container exited during start-up")
            time.sleep(5)
    raise RuntimeError("vLLM did not become ready")


def stop_vllm():
    subprocess.run(["podman", "stop", "-t", "20", "dl-vllm"], capture_output=True)


# ---------------------------------------------------------------- rollouts + reward
def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7:j - 1] if d == 0 else None


def is_correct(bench, final, gold):
    if not final:
        return False
    try:
        from math_verify import parse, verify
        if verify(parse("$" + gold + "$"), parse(final)):
            return True
    except Exception:
        pass
    b = boxed(final)
    if b is None:
        return False
    norm = lambda x: re.sub(r"[\s,$]|\\!|\\text\{[^}]*\}|\\left|\\right", "", str(x)).rstrip(".").replace("dfrac", "frac")
    return norm(b) == norm(gold)


def rollout(port, item, seed, cap, gen):
    body = {"model": "policy", "messages": [{"role": "user", "content": PROMPT.format(q=item["question"])}],
            "seed": seed, "max_tokens": cap, "temperature": gen["temperature"], "top_p": gen["top_p"],
            "top_k": gen["top_k"], "presence_penalty": gen["presence_penalty"],
            "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False,
            "return_token_ids": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    c = r["choices"][0]
    text = c["message"].get("content") or ""
    closed = "</think>" in text
    final = text.split("</think>", 1)[1] if closed else ""
    ok = closed and c["finish_reason"] == "stop" and is_correct(item["bench"], final, item["answer"])
    n = r["usage"]["completion_tokens"]
    return {"id": item["id"], "bench": item["bench"], "seed": seed, "tokens": n, "finish": c["finish_reason"],
            "closed": closed, "correct": ok, "reward": (1.0 + 0.2 * (1 - n / cap) if ok else 0.0) + (0.25 if closed and c["finish_reason"] == "stop" else 0.0),
            "prompt_ids": r.get("prompt_token_ids"), "completion_ids": c.get("token_ids")}


# ---------------------------------------------------------------- training
def chunk_lp(head, h, t):
    return torch.log_softmax(head(h).float(), -1).gather(-1, t[:, None]).sum()


def gate_params(model):
    out = {}
    for name, p in model.named_parameters():
        m = GATE_RE.search(name)
        if m:
            out[name] = p
        p.requires_grad_(bool(m))
    return out


def export(parent_dir, master, model_param_names, out_dir):
    """Copy parent dir; replace the gate tensors in model.safetensors by bf16/f32(master) at the parent's dtype."""
    assert not os.path.exists(out_dir), f"{out_dir} exists; refusing to overwrite"
    shutil.copytree(parent_dir, out_dir, ignore=shutil.ignore_patterns("model.safetensors"))
    sd = load_file(f"{parent_dir}/model.safetensors")
    with safe_open(f"{parent_dir}/model.safetensors", "pt") as f:
        meta = f.metadata()
    changed = 0
    for pname, ckey in model_param_names.items():
        sd[ckey] = master[pname].detach().to(sd[ckey].dtype).cpu().contiguous()
        changed += 1
    save_file(sd, f"{out_dir}/model.safetensors", metadata=meta)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gen", default="evaluation.json")
    ap.add_argument("--steps", type=int, default=15); ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--cap", type=int, default=6144)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261124); ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--util", type=float, default=0.30); ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--kill-truncation", type=float, default=None, help="stop after the first eval if truncation exceeds this")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--pool", default=None, help="practice problems jsonl (e.g. filtered to ones the parent can solve)")
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--max-neg", type=int, default=4, help="failures trained per group (importance-weighted)")
    ap.add_argument("--train-bench", default=None, help="restrict practice problems to one source (e.g. gsm8k)"); ap.add_argument("--eval-cap", type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    gen = json.load(open(a.gen))["generation"]
    train = [json.loads(l) for l in open(a.pool or f"{a.data}/train.jsonl")]
    if a.train_bench:
        train = [r for r in train if r["bench"] == a.train_bench]
    held = [json.loads(l) for l in open(f"{a.data}/heldout.jsonl")]
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.parent)
    model = AutoModelForCausalLM.from_pretrained(a.parent, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    gates = gate_params(model)
    # map model param names -> checkpoint keys (checkpoint uses model.language_model.layers...)
    with safe_open(f"{a.parent}/model.safetensors", "pt") as f:
        ckeys = list(f.keys())
    name_map = {}
    for pname in gates:
        m = GATE_RE.search(pname)
        suffix = f"layers.{m.group(1)}.linear_attn.{m.group(2)}"
        hits = [k for k in ckeys if k.startswith("model.language_model.") and k.endswith(suffix)]
        assert len(hits) == 1, (pname, hits)
        name_map[pname] = hits[0]
    with safe_open(f"{a.parent}/model.safetensors", "pt") as f:   # fp32 master from the checkpoint itself
        master = {n: f.get_tensor(name_map[n]).float().to(dev) for n in gates}
    with torch.no_grad():
        for n, p in gates.items():
            p.copy_(master[n].to(p.dtype))
    opt_state = {n: (torch.zeros_like(v), torch.zeros_like(v)) for n, v in master.items()}
    log(a.out, f"trainable gate tensors={len(gates)} params={sum(v.numel() for v in master.values())}")

    def evaluate(step, port):
        rows = []
        with cf.ThreadPoolExecutor(a.conc) as ex:
            futs = [ex.submit(rollout, port, it, a.seed + 1000 * i, a.eval_cap, gen) for i, it in enumerate(held)]   # = think_probe sample 0 seeds
            rows = [f.result() for f in futs]
        n = len(rows)
        s = {"step": step, "n": n, "correct": sum(r["correct"] for r in rows) / n,
             "closed": sum(r["closed"] for r in rows) / n,
             "truncated": sum(r["finish"] == "length" for r in rows) / n,
             "mean_tokens": sum(r["tokens"] for r in rows) / n}
        with open(f"{a.out}/eval.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        with open(f"{a.out}/eval-items-step{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "bench", "seed", "tokens", "finish", "closed", "correct")}) + "\n")
        log(a.out, f"EVAL step {step}: " + json.dumps(s))
        return s

    policy = f"{a.out}/policy-step0"
    export(a.parent, master, name_map, policy)
    for step in range(a.steps + 1):
        t = start_vllm(policy, a.port, a.util, a.max_seqs)
        log(a.out, f"step {step}: vLLM ready in {t:.0f}s")
        if step > 0 and (step % a.eval_every == 0 or step == a.steps):   # step 0 = runs/failure-orig.jsonl
            ev = evaluate(step, a.port)
            if a.kill_truncation is not None and step == a.eval_every and ev["truncated"] > a.kill_truncation:
                log(a.out, f"KILL RULE: truncation {ev['truncated']:.3f} > {a.kill_truncation} at step {step}; stopping")
                stop_vllm(); break
        if step == a.steps:
            stop_vllm(); break
        batch = rng.sample(train, min(a.prompts, len(train)))
        jobs = [(it, a.seed + 100000 * step + 100 * i + g) for i, it in enumerate(batch) for g in range(a.group)]
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            res = list(ex.map(lambda j: rollout(a.port, j[0], j[1], a.cap, gen), jobs))
        stop_vllm()
        gen_s = time.time() - t0
        # advantages
        groups = {}
        for r in res:
            groups.setdefault(r["id"], []).append(r)
        samples = []
        for rs in groups.values():
            rw = torch.tensor([r["reward"] for r in rs])
            if rw.std() < 1e-6:
                continue
            adv = (rw - rw.mean()) / (rw.std() + 1e-4)
            pos = [(r, float(x)) for r, x in zip(rs, adv) if x > 0]
            neg = [(r, float(x)) for r, x in zip(rs, adv) if x <= 0]
            keep = rng.sample(neg, min(len(neg), max(a.max_neg, len(pos))))
            w = len(neg) / max(1, len(keep))          # importance weight keeps the group gradient unbiased
            samples += pos + [(r, x * w) for r, x in keep]
        stats = {"step": step, "gen_s": round(gen_s), "rollouts": len(res),
                 "reward": sum(r["reward"] for r in res) / len(res),
                 "correct": sum(r["correct"] for r in res) / len(res),
                 "closed": sum(r["closed"] for r in res) / len(res),
                 "truncated": sum(r["finish"] == "length" for r in res) / len(res),
                 "mean_tokens": sum(r["tokens"] for r in res) / len(res),
                 "useful_groups": len({r["id"] for r, _ in samples})}
        # policy gradient, one update
        t0 = time.time()
        model.train()
        for p in gates.values():
            p.grad = None
        for r, adv in samples:
            ids = torch.tensor([r["prompt_ids"] + r["completion_ids"]], device=dev)
            np_ = len(r["prompt_ids"])
            hidden = model.model(input_ids=ids).last_hidden_state[0, np_ - 1:-1]
            tgt = ids[0, np_:]
            lp_sum = 0.0
            for c in range(0, hidden.shape[0], 1024):   # chunked, recomputed in backward: no full logits
                lp_sum = lp_sum + torch.utils.checkpoint.checkpoint(chunk_lp, model.lm_head, hidden[c:c + 1024], tgt[c:c + 1024], use_reentrant=False)
            loss = -adv * lp_sum / a.cap / max(1, len(samples))
            loss.backward()
            del hidden, lp_sum, loss
        gn = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in gates.values() if p.grad is not None))
        scale = min(1.0, a.clip / (gn + 1e-12))
        b1, b2, eps = 0.9, 0.99, 1e-8
        with torch.no_grad():
            for n, p in gates.items():
                if p.grad is None:
                    continue
                g = p.grad.float() * scale
                m1, m2 = opt_state[n]
                m1.mul_(b1).add_(g, alpha=1 - b1); m2.mul_(b2).addcmul_(g, g, value=1 - b2)
                mh, vh = m1 / (1 - b1 ** (step + 1)), m2 / (1 - b2 ** (step + 1))
                master[n].add_(-a.lr * mh / (vh.sqrt() + eps))
                p.copy_(master[n].to(p.dtype))
        stats.update({"train_s": round(time.time() - t0), "grad_norm": gn, "clip_scale": scale})
        log(a.out, "STEP " + json.dumps(stats))
        with open(f"{a.out}/steps.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
        policy = f"{a.out}/policy-step{step + 1}"
        export(a.parent, master, name_map, policy)
    log(a.out, "done")


if __name__ == "__main__":
    main()
