import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, tuned, out, alpha = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
for f in tuned.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"):
        shutil.copy(f, out / f.name)
tensors, moved = {}, 0
with safe_open(str(base / "model.safetensors"), "pt") as fb, safe_open(str(tuned / "model.safetensors"), "pt") as ft:
    for k in ft.keys():
        t = ft.get_tensor(k)
        b = fb.get_tensor(k)
        d = t.float() - b.float()
        if d.abs().max() > 0:
            moved += 1
        tensors[k] = (t.float() + (alpha - 1.0) * d).to(t.dtype)
save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
print(f"{out.name}: alpha={alpha} tensors moved={moved} of {len(tensors)}")
