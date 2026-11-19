"""Model extrapolation (ExPO): OUT = G + a * (G - P) on every tensor where G and P differ (fp32 math, back to G's dtype).
usage: expo.py G_DIR P_DIR A OUT_DIR"""
import sys, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
g, p, a, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
G = safe_open(os.path.join(g, 'model.safetensors'), 'pt'); P = safe_open(os.path.join(p, 'model.safetensors'), 'pt')
res = {}; n = 0
for k in G.keys():
    x = G.get_tensor(k); y = P.get_tensor(k)
    if not torch.equal(x, y): x = (x.float() + a * (x.float() - y.float())).to(x.dtype); n += 1
    res[k] = x.contiguous()
os.makedirs(out, exist_ok=True)
for f in os.listdir(g):
    if f != 'model.safetensors': shutil.copy(os.path.join(g, f), os.path.join(out, f))
save_file(res, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'}); print('extrapolated tensors', n, 'alpha', a)
