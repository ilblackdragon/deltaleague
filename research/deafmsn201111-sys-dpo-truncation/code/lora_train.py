"""Step 3a: LoRA SFT on selected traces (completion-only loss).

Works on CUDA (fp16 autocast, for T4) and on MPS/CPU (fp32, for smoke tests).
Saves lora.safetensors with tensors "<module>.A" [r, in] and "<module>.B" [out, r]
plus lora.json with rank/alpha/target names, independent of PEFT internals.

Usage: python lora_train.py --model DIR --data sft.jsonl --out OUT_DIR [--limit N] [--max-len 6656]
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.utils.checkpoint
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_RE = r".*layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--max-len", type=int, default=6656)
ap.add_argument("--truncate", action="store_true", help="truncate long examples instead of skipping (tests only)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-minutes", type=float, default=0, help="stop early and save after this many minutes")
ap.add_argument("--loss-chunk", type=int, default=1024)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = dev == "cuda"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)

# ---- minimal LoRA (no PEFT dependency in the saved format) ----
class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


targets = []
for name, mod in list(model.named_modules()):
    if isinstance(mod, torch.nn.Linear) and re.match(TARGET_RE, name):
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, LoRALinear(mod, args.rank, args.alpha))
        targets.append(name)
lora_params = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
print(f"device {dev}; LoRA on {len(targets)} modules; trainable {sum(p.numel() for p in lora_params) / 1e6:.2f}M", flush=True)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

# ---- data ----
examples = []
skipped = 0
for line in open(args.data):
    r = json.loads(line)
    p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    c = tok(r["completion"], add_special_tokens=False)["input_ids"]
    ids = p + c
    if len(ids) > args.max_len:
        if not args.truncate:
            skipped += 1
            continue
        ids = ids[:args.max_len]
    labels = [-100] * len(p) + ids[len(p):]
    examples.append((ids, labels))
random.shuffle(examples)
if args.limit:
    examples = examples[:args.limit]
n_steps = max(1, int(len(examples) * args.epochs) // args.grad_accum)
print(f"examples {len(examples)} (skipped {skipped} too long); optimizer steps {n_steps}", flush=True)

opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
warm = max(1, n_steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

model.train()
t0 = time.time()
step, micro, run_loss, run_tok = 0, 0, 0.0, 0
order = [examples[i % len(examples)] for i in range(n_steps * args.grad_accum)]
def chunk_loss(h, w, t):
    logits = torch.nn.functional.linear(h, w).float()
    return torch.nn.functional.cross_entropy(logits, t, ignore_index=-100, reduction="sum")


lm_w = model.get_output_embeddings().weight
for ids, labels in order:
    if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
        print("time budget reached, stopping early", flush=True)
        break
    x = torch.tensor([ids], device=dev)
    tgt = torch.tensor(labels[1:], device=dev)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        h = model.model(input_ids=x).last_hidden_state[0, :-1]
        # only positions that predict completion tokens, in chunks, recomputed in backward
        keep = (tgt != -100).nonzero().squeeze(-1)
        h, tgt = h[keep], tgt[keep]
        loss_sum = 0.0
        for c in range(0, h.size(0), args.loss_chunk):
            loss_sum = loss_sum + torch.utils.checkpoint.checkpoint(
                chunk_loss, h[c:c + args.loss_chunk], lm_w, tgt[c:c + args.loss_chunk], use_reentrant=False)
    n_tok = int(tgt.numel())
    # normalise by a fixed 1000 tokens per micro-batch so long and short traces weigh by length
    scaler.scale(loss_sum / 1000.0 / args.grad_accum).backward()
    run_loss += float(loss_sum.detach()); run_tok += n_tok
    micro += 1
    if micro % args.grad_accum == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 5 == 0 or step == n_steps:
            el = time.time() - t0
            print(f"step {step}/{n_steps} loss/token {run_loss / max(1, run_tok):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} elapsed {el / 60:.1f} min eta {el / step * (n_steps - step) / 60:.1f} min",
                  flush=True)
            run_loss, run_tok = 0.0, 0

# ---- save ----
tensors = {}
for name, mod in model.named_modules():
    if isinstance(mod, LoRALinear):
        tensors[f"{name}.A"] = mod.A.detach().float().cpu().contiguous()
        tensors[f"{name}.B"] = mod.B.detach().float().cpu().contiguous()
save_file(tensors, str(out / "lora.safetensors"))
(out / "lora.json").write_text(json.dumps({"rank": args.rank, "alpha": args.alpha, "targets": targets,
                                           "args": vars(args), "examples": len(examples), "steps": n_steps}, indent=1))
print(f"saved {len(tensors)} tensors to {out}; total {(time.time() - t0) / 60:.1f} min", flush=True)
