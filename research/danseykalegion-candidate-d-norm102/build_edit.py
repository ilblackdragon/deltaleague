"""Build Candidate E: one final-RMSNorm edit on public parent 344c8376.

Parent 344c8376's public lineage does not change this tensor. The edit scales
its effective RMSNorm gain, `(1 + w)`, by 1.02. No other tensor changes.
"""
import sys

import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) != 3:
    raise SystemExit("usage: build_edit.py PARENT.safetensors OUT.safetensors")

weights = load_file(sys.argv[1])
name = "model.language_model.norm.weight"
weights[name] = ((1.0 + weights[name].float()) * 1.02 - 1.0).to(torch.bfloat16)
save_file(weights, sys.argv[2], metadata={"format": "pt"})
