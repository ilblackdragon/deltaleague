"""Option B: DPO + NLL on chosen, LoRA over a fixed base. Same output format as lora_train.py.

Reference log-probs come from the base before training (LoRA B starts at zero), so no second model.
Usage: python dpo_train.py --model DIR --data pairs.jsonl --out DIR [--beta .1 --lr 2e-5 --nll 0.2]
"""
import argparse, json, math, random, re, time
from pathlib import Path
import torch
import torch.utils.checkpoint
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_RE = r".*layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
ap = argparse.ArgumentParser()
for a, t, d in [("--model", str, None), ("--data", str, None), ("--out", str, None), ("--rank", int, 16), ("--alpha", float, 32),
                ("--lr", float, 2e-5), ("--beta", float, 0.1), ("--nll", float, 0.2), ("--grad-accum", int, 8),
                ("--limit", int, 0), ("--seed", int, 0), ("--max-minutes", float, 0), ("--loss-chunk", int, 1024),
                ("--max-len", int, 0), ("--epochs", int, 1)]:
    ap.add_argument(a, type=t, default=d)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
amp = dev == "cuda"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
model.config.use_cache = False
for p in model.parameters(): p.requires_grad_(False)


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


targets = []
for name, mod in list(model.named_modules()):
    if isinstance(mod, torch.nn.Linear) and re.match(TARGET_RE, name):
        pn, ch = name.rsplit(".", 1)
        setattr(model.get_submodule(pn), ch, LoRALinear(mod, args.rank, args.alpha)); targets.append(name)
params = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()
lm_w = model.get_output_embeddings().weight


def enc(prompt, comp):
    p = tok(prompt, add_special_tokens=False)["input_ids"]; c = tok(comp, add_special_tokens=False)["input_ids"]
    if args.max_len: c = c[:max(1, args.max_len - len(p))]
    return p + c, len(p)


data = []
for line in open(args.data):
    r = json.loads(line)
    data.append((enc(r["prompt"], r["chosen"]), enc(r["prompt"], r["rejected"])))
if args.limit: data = data[:args.limit]


def chunk_lp(h, w, t):
    return torch.log_softmax(torch.nn.functional.linear(h, w).float(), -1).gather(-1, t[:, None]).sum()


def seq_logp(ids, plen):
    x = torch.tensor([ids], device=dev); t = torch.tensor(ids[plen:], device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        h = model.model(input_ids=x).last_hidden_state[0, plen - 1:-1]
        s = 0.0
        for c in range(0, h.size(0), args.loss_chunk):
            s = s + torch.utils.checkpoint.checkpoint(chunk_lp, h[c:c + args.loss_chunk], lm_w,
                                                      t[c:c + args.loss_chunk], use_reentrant=False)
    return s, t.numel()


t0 = time.time()
ref = []
with torch.no_grad():
    for i, (c, r) in enumerate(data):
        ref.append((float(seq_logp(*c)[0]), float(seq_logp(*r)[0])))
        if i % 50 == 0: print(f"ref {i}/{len(data)} {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"pairs {len(data)}; LoRA on {len(targets)} modules; ref pass {(time.time() - t0) / 60:.1f} min", flush=True)

order = [i for _ in range(args.epochs) for i in random.sample(range(len(data)), len(data))]
n_steps = max(1, len(order) // args.grad_accum)
order = order[:n_steps * args.grad_accum]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
warm = max(1, n_steps // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
scaler = torch.amp.GradScaler("cuda", enabled=amp)
st = dict(loss=0.0, acc=0.0, margin=0.0, dc=0.0, dr=0.0, n=0)
step = 0
t1 = time.time()
for m, i in enumerate(order):
    if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
        print("time budget reached, stopping early", flush=True); break
    (c, r), (rc, rr) = data[i], ref[i]
    pc, nc = seq_logp(*c); pr, _ = seq_logp(*r)
    margin = args.beta * ((pc - rc) - (pr - rr))
    loss = -torch.nn.functional.logsigmoid(margin) + args.nll * (-pc / nc)
    scaler.scale(loss / args.grad_accum).backward()
    st["loss"] += float(loss.detach()); st["acc"] += float(margin.detach() > 0); st["margin"] += float(margin.detach())
    st["dc"] += float(pc.detach()) - rc; st["dr"] += float(pr.detach()) - rr; st["n"] += 1
    if (m + 1) % args.grad_accum == 0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        if step % 5 == 0 or step == n_steps or step == 1:
            n = st["n"]
            print(f"step {step}/{n_steps} loss {st['loss'] / n:.4f} acc {st['acc'] / n:.2f} margin {st['margin'] / n:.3f} "
                  f"d_chosen {st['dc'] / n:.2f} d_rejected {st['dr'] / n:.2f} elapsed {(time.time() - t1) / 60:.1f} min", flush=True)
            st = dict(loss=0.0, acc=0.0, margin=0.0, dc=0.0, dr=0.0, n=0)

tensors = {}
for name, mod in model.named_modules():
    if isinstance(mod, LoRALinear):
        tensors[f"{name}.A"] = mod.A.detach().float().cpu().contiguous()
        tensors[f"{name}.B"] = mod.B.detach().float().cpu().contiguous()
save_file(tensors, str(out / "lora.safetensors"))
(out / "lora.json").write_text(json.dumps({"rank": args.rank, "alpha": args.alpha, "targets": targets, "args": vars(args),
                                           "pairs": len(data), "steps": step}, indent=1))
print(f"saved to {out}; total {(time.time() - t0) / 60:.1f} min", flush=True)
