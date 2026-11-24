"""Port a task vector to a new base: OUT = NEW_BASE + (MODEL - OLD_BASE), in float64, stored in NEW_BASE's dtypes.
Tensors where MODEL equals OLD_BASE are copied from NEW_BASE unchanged.
usage: python port_delta.py OLD_BASE.safetensors MODEL.safetensors NEW_BASE.safetensors OUT.safetensors"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
old, model, new, out = sys.argv[1:5]
O, M, N = safe_open(old, 'pt'), safe_open(model, 'pt'), safe_open(new, 'pt')
assert set(O.keys()) == set(M.keys()) == set(N.keys())
res, changed = {}, 0
for k in sorted(N.keys()):
    o, m, n = O.get_tensor(k), M.get_tensor(k), N.get_tensor(k)
    if torch.equal(o, m):
        res[k] = n
    else:
        res[k] = (n.double() + (m.double() - o.double())).to(n.dtype); changed += 1
save_file(res, out, metadata={'format': 'pt'})
print(f'{changed} tensors carry the ported delta -> {out}')
