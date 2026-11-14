"""Task-vector extrapolation: OUT = PARENT + LAMBDA * (MODEL - PARENT), computed in float64 and stored in each
tensor's original dtype. Tensors equal in PARENT and MODEL are copied unchanged.
usage: python extrapolate.py PARENT.safetensors MODEL.safetensors LAMBDA OUT.safetensors"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
parent, model, lam, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
P, M = safe_open(parent, 'pt'), safe_open(model, 'pt')
assert set(P.keys()) == set(M.keys())
res, changed = {}, 0
for k in sorted(P.keys()):
    p, m = P.get_tensor(k), M.get_tensor(k)
    if torch.equal(p, m):
        res[k] = p
    else:
        res[k] = (p.double() + lam * (m.double() - p.double())).to(p.dtype); changed += 1
save_file(res, out, metadata={'format': 'pt'})
print(f'lambda {lam}: {changed} tensors extrapolated -> {out}')
