"""Full-parameter SFT of the student on selected teacher traces (completion-only loss), saved in the parent's
tensor layout and dtypes. Tensors the HF model does not load are copied through unchanged (as in #58's grpo.py).
Optionally also writes a linear blend parent + alpha * (trained - parent).

usage: python train_sft.py PARENT_DIR train.jsonl OUT_DIR [--lr 5e-6] [--epochs 1] [--accum 16] [--max-len 5120] [--blend 0.5]
"""
import argparse, json, math, random, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser(); ap.add_argument('parent'); ap.add_argument('data'); ap.add_argument('out')
ap.add_argument('--lr', type=float, default=5e-6); ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--accum', type=int, default=16); ap.add_argument('--max-len', type=int, default=5120)
ap.add_argument('--blend', type=float, default=0.5); ap.add_argument('--seed', type=int, default=20261112)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
from transformers import AutoModelForImageTextToText, AutoTokenizer

parent = Path(a.parent); out = Path(a.out)
with safe_open(str(parent / 'model.safetensors'), 'pt') as f:
    dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
dtypes = {k: {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}[v] for k, v in dtypes.items()}
tok = AutoTokenizer.from_pretrained(a.parent)
model = AutoModelForImageTextToText.from_pretrained(a.parent, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.config.use_cache = False; model.train()
held = set(model.state_dict())
with safe_open(str(parent / 'model.safetensors'), 'pt') as f:
    frozen = {k: f.get_tensor(k) for k in dtypes if k not in held}

ex = []
for l in open(a.data):
    r = json.loads(l)
    p = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    pi = tok(p, add_special_tokens=False)['input_ids']
    ci = tok(r['completion'] + '<|im_end|>', add_special_tokens=False)['input_ids']
    if len(pi) + len(ci) <= a.max_len:
        ex.append((pi, ci))
print(f'{len(ex)} examples, {sum(len(c) for _, c in ex)} completion tokens', flush=True)

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
steps = math.ceil(len(ex) * a.epochs / a.accum); warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
t0, step, run, ntok = time.time(), 0, 0.0, 0
for ep in range(a.epochs):
    random.shuffle(ex)
    for i, (pi, ci) in enumerate(ex):
        ids = torch.tensor([pi + ci], device='cuda'); labels = torch.tensor([[-100] * len(pi) + ci], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=labels).loss / a.accum
        loss.backward(); run += loss.item(); ntok += len(ci)
        if (i + 1) % a.accum == 0 or i + 1 == len(ex):
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0 or step == steps:
                print(f'step {step}/{steps} loss {run / 10 if step % 10 == 0 else run:.4f} lr {sched.get_last_lr()[0]:.2e} '
                      f'{ntok / (time.time() - t0):.0f} tok/s {(time.time() - t0) / 60:.1f} min', flush=True)
                run = 0.0

cur = model.state_dict()
trained = {k: frozen[k] if k in frozen else cur[k].detach().cpu().to(dt).contiguous() for k, dt in dtypes.items()}


def write(name, tensors):
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    for f in parent.iterdir():
        if f.is_file() and f.name not in ('model.safetensors', 'delta-model.json'):
            shutil.copy(f, d / f.name)
    save_file(tensors, str(d / 'model.safetensors'), metadata={'format': 'pt'})
    print('saved', d, flush=True)


write('sft', trained)
if a.blend:
    with safe_open(str(parent / 'model.safetensors'), 'pt') as f:
        blend = {k: (f.get_tensor(k).float() + a.blend * (trained[k].float() - f.get_tensor(k).float())).to(dtypes[k])
                 for k in dtypes}
    write(f'blend{int(a.blend * 100)}', blend)
print('SFT-DONE', flush=True)
