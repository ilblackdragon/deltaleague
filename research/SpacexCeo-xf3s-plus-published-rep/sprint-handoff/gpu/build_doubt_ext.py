"""Extend xf3s's doubt-direction projection to more writer layers on top of b8af1dbc (idempotent on layers 0-13).
usage: build_doubt_ext.py PARENT_DIR OUT_DIR DIRECTION_JSON LAYERS(comma list) [ALPHA]
W <- W - alpha * d (d . W) for self_attn.o_proj / linear_attn.out_proj / mlp.down_proj of the given layers."""
import sys, os, json, shutil, hashlib, torch
from safetensors.torch import load_file, save_file
par, out, dj, layers = sys.argv[1:5]; alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
os.makedirs(out, exist_ok=True)
for f in os.listdir(par):
    if f != 'model.safetensors': shutil.copy(os.path.join(par, f), os.path.join(out, f))
w = load_file(os.path.join(par, 'model.safetensors'))
d = torch.tensor(json.load(open(dj))['values'], dtype=torch.float32); d = d / d.norm()
ch = []
for L in [int(x) for x in layers.split(',')]:
    for part in ('self_attn.o_proj', 'linear_attn.out_proj', 'mlp.down_proj'):
        k = f'model.language_model.layers.{L}.{part}.weight'
        if k in w:
            W = w[k].float(); W -= alpha * torch.outer(d, d @ W); w[k] = W.to(torch.bfloat16); ch.append(k)
save_file(w, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
print(len(ch), 'tensors', ch[:3], '...'); print('sha256', hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest())
