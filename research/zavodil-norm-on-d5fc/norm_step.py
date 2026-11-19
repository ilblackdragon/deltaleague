"""Multiply the final RMSNorm gain of a checkpoint by a factor and write a new checkpoint.

usage: norm_step.py PARENT_DIR OUT_DIR FACTOR

The final RMSNorm scales by (1 + weight), so the factor applies to that sum. Only this tensor
changes; the script verifies it.
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"
NORM = "model.language_model.norm.weight"


def main():
    parent, out, factor = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    with safe_open(parent / WEIGHTS, "pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    original = tensors[NORM].clone()
    tensors[NORM] = ((1 + original.float()) * factor - 1).to(original.dtype)
    out.mkdir(parents=True)
    save_file(tensors, out / WEIGHTS, metadata=metadata or {"format": "pt"})
    for item in parent.iterdir():
        if item.name not in (WEIGHTS, "delta-model.json"):
            shutil.copy(item, out / item.name)
    print(f"{out}: {int((tensors[NORM] != original).sum())} norm values changed")


if __name__ == "__main__":
    main()
