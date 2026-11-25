"""Uniform weight soup of checkpoints with identical keys. usage: soup.py OUT_DIR CKPT1 CKPT2 [...] (configs copied from CKPT1)"""
import sys, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
out, cks = sys.argv[1], sys.argv[2:]
F = [safe_open(os.path.join(c, 'model.safetensors'), 'pt') for c in cks]; res = {}
for k in F[0].keys():
    ts = [f.get_tensor(k) for f in F]
    res[k] = ts[0] if all(torch.equal(ts[0], t) for t in ts[1:]) else (sum(t.float() for t in ts) / len(ts)).to(ts[0].dtype).contiguous()
os.makedirs(out, exist_ok=True)
for f in os.listdir(cks[0]):
    if f != 'model.safetensors': shutil.copy(os.path.join(cks[0], f), os.path.join(out, f))
save_file(res, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'}); print('soup of', len(cks))
