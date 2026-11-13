import hashlib, json, os
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SRC = os.path.expanduser('~/workspace/delta-league/experiment-v6/model/model.safetensors')
DST = os.path.expanduser('~/workspace/delta-league/build/edited.safetensors')
TARGET = 'model.language_model.norm.weight'
FACTOR = 1.05

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

with safe_open(SRC, framework='pt') as f:
    meta = f.metadata()
print('original metadata:', meta)

sd = load_file(SRC)
assert TARGET in sd, TARGET
orig_t = sd[TARGET].clone()
sd[TARGET] = (sd[TARGET].float() * FACTOR).to(sd[TARGET].dtype)
save_file(sd, DST, metadata=meta)

print('orig size:', os.path.getsize(SRC))
print('edit size:', os.path.getsize(DST))
print('orig sha256:', sha256(SRC))
print('edit sha256:', sha256(DST))

# tensor-by-tensor verification
a = load_file(SRC)
b = load_file(DST)
assert set(a.keys()) == set(b.keys()), 'key sets differ'
diffs = []
for k in a:
    if not torch.equal(a[k], b[k]):
        diffs.append(k)
print('tensors changed:', diffs)
t = b[TARGET].float()
ref = a[TARGET].float() * FACTOR
max_err = (t - ref).abs().max().item()
rel = max_err / ref.abs().max().item()
print(f'target max |edited - orig*1.05| = {max_err:.6g} (rel {rel:.3g}); bf16 ulp scale ok if rel < ~0.008')
exact_others = all(torch.equal(a[k], b[k]) for k in a if k != TARGET)
print('all other tensors bit-identical:', exact_others)
print('target orig[:4] :', a[TARGET].float().flatten()[:4].tolist())
print('target edit[:4] :', b[TARGET].float().flatten()[:4].tolist())
