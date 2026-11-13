"""Sanity check: edited logits ~= ALPHA * original logits; greedy argmax unchanged.
Usage: python check_logits.py WORKSPACE_DIR   (uses WORKSPACE/model and WORKSPACE/edited.safetensors)"""
import json, shutil, sys, tempfile
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
ws = Path(sys.argv[1]).resolve()
prompts = ['Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?',
           'The capital of France is', 'def fibonacci(n):']
tok = AutoTokenizer.from_pretrained(ws / 'model')
def logits(model_dir):
    m = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).eval()
    out = []
    with torch.inference_mode():
        for p in prompts:
            ids = tok(p, return_tensors='pt')
            out.append(m(**ids).logits[0].float())
    del m
    return out
base = logits(ws / 'model')
with tempfile.TemporaryDirectory() as tmp:
    for f in (ws / 'model').iterdir():
        if f.name != 'model.safetensors': shutil.copy(f, tmp)
    shutil.copy(ws / 'edited.safetensors', Path(tmp) / 'model.safetensors')
    edit = logits(tmp)
res = []
for p, b, e in zip(prompts, base, edit):
    slope = float((b * e).sum() / (b * b).sum())
    resid = float((e - slope * b).norm() / e.norm())
    agree = float((b.argmax(-1) == e.argmax(-1)).float().mean())
    tb = torch.softmax(b[-1], -1); te = torch.softmax(e[-1], -1)
    ent = lambda q: float(-(q * q.clamp_min(1e-30).log()).sum())
    res.append({'prompt': p[:40], 'positions': b.shape[0], 'fitted_scale': slope, 'relative_residual': resid,
                'argmax_agreement': agree, 'last_pos_entropy_nats': [ent(tb), ent(te)],
                'last_pos_top1_prob': [float(tb.max()), float(te.max())]})
print(json.dumps(res, indent=1))
