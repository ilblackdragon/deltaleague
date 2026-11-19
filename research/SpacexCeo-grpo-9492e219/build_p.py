"""Bake a per-layer direction ablation into a parent: for each layer L in the band and each writer (self_attn.o_proj /
linear_attn.out_proj / mlp.down_proj): W <- W - a * u_L (u_L^T W)  (fp32 math, back to the parent dtype). u_L = DIRS[f'{KEY}_L{L}'].
usage: build_p.py PARENT_DIR DIRS.pt KEY L0 L1 ALPHA OUT_DIR   (band = L0..L1 inclusive). Writes OUT_DIR and OUT_DIR.build.json."""
import sys, os, json, shutil, hashlib, torch
from safetensors.torch import load_file, save_file
parent, dirs, key, l0, l1, a, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
D = torch.load(dirs); W = load_file(os.path.join(parent, 'model.safetensors')); shapes = {k: (v.shape, v.dtype) for k, v in W.items()}
changed = {}
for L in range(l0, l1 + 1):
    u = D[f'{key}_L{L}'].float(); u = u / u.norm()
    for part in ('self_attn.o_proj', 'linear_attn.out_proj', 'mlp.down_proj'):
        k = f'model.language_model.layers.{L}.{part}.weight'
        if k in W:
            M = W[k].float(); M -= a * torch.outer(u, u @ M); W[k] = M.to(W[k].dtype).contiguous()
            changed[k] = hashlib.sha256(W[k].view(torch.uint16).numpy().tobytes()).hexdigest()
assert {k: (v.shape, v.dtype) for k, v in W.items()} == shapes and changed
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
json.dump({'parent': parent, 'dirs': dirs, 'key': key, 'band': [l0, l1], 'alpha': a, 'tensors': changed}, open(out.rstrip('/') + '.build.json', 'w'), indent=1)
print(len(changed), 'tensors changed'); [print(changed[k][:16], k) for k in sorted(changed)]
