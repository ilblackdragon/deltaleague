"""Write a float32 copy of a checkpoint directory for fast, memory-shared CPU inference.

All floating tensors are cast to float32; other files are symlinked. The original
(BF16) directory is never modified. Only the fp32 copy is used for local inference;
patches are always built from BF16 tensors.

usage: python to_fp32.py SRC_MODEL_DIR DST_MODEL_DIR
"""
import os, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = Path(sys.argv[1]).expanduser().resolve(), Path(sys.argv[2]).expanduser().resolve()
dst.mkdir(parents=True, exist_ok=True)
tensors = {}
with safe_open(src / "model.safetensors", framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        t = f.get_tensor(k)
        tensors[k] = t.float() if t.is_floating_point() else t
save_file(tensors, dst / "model.safetensors", metadata=meta or {"format": "pt"})
for p in src.iterdir():
    if p.name not in ("model.safetensors", "delta-model.json") and not (dst / p.name).exists():
        os.symlink(p, dst / p.name)
print(f"wrote {dst / 'model.safetensors'}: {len(tensors)} tensors, {(dst / 'model.safetensors').stat().st_size / 1e9:.2f} GB")
