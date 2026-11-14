"""Uniform weight average ("model soup") of several checkpoints with identical tensor names.
Averaged in float64, stored in each tensor's original dtype. usage: soup.py OUT.safetensors IN1 IN2 [IN3 ...]"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
out, ins = sys.argv[1], sys.argv[2:]
fs = [safe_open(p, 'pt') for p in ins]
keys = set(fs[0].keys()); assert all(set(f.keys()) == keys for f in fs)
res = {}
for k in sorted(keys):
    ts = [f.get_tensor(k) for f in fs]
    res[k] = ts[0] if all(torch.equal(ts[0], t) for t in ts[1:]) else (sum(t.double() for t in ts) / len(ts)).to(ts[0].dtype)
save_file(res, out, metadata={'format': 'pt'})
print('souped', len(ins), 'checkpoints ->', out)
