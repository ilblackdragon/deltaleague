"""Scale the final RMSNorm gain of a BF16 checkpoint: (1 + w) -> gain * (1 + w).

Equivalent to dividing the sampling temperature by `gain` for every token. Only
model.language_model.norm.weight changes; every other tensor is copied bit for bit.

usage: python edit_norm.py PARENT.safetensors OUT.safetensors GAIN
"""
import hashlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

parent, out, gain = sys.argv[1], sys.argv[2], float(sys.argv[3])
KEY = "model.language_model.norm.weight"
tensors = {}
with safe_open(parent, framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
w = tensors[KEY]
new = ((1 + w.double()) * gain - 1).to(w.dtype)
realized = (1 + new.double()) / (1 + w.double())
tensors[KEY] = new
save_file(tensors, out, metadata=meta)
print(f"{KEY}: requested gain {gain}, realized after {w.dtype} rounding: mean {realized.mean():.5f}, min {realized.min():.5f}, max {realized.max():.5f}")
print("tensor sha256:", hashlib.sha256(new.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
print("file sha256:  ", hashlib.sha256(open(out, "rb").read()).hexdigest())
