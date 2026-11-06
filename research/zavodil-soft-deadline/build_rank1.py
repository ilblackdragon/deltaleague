"""Apply rank-1 edits to layer 23 down_proj and write a new checkpoint.

usage: build_rank1.py BASE_DIR DIRECTIONS.npz OUT_DIR EDIT [EDIT ...]
  EDIT = phase_boxed:DELTA   adds DELTA * v_boxed * u_phase^T
         pos_close:DELTA     adds DELTA * v_close * u_pos^T

down_proj maps the 3584-d MLP intermediate to the 1024-d residual, so for input x the
residual gains DELTA * (u . x) * v. Only that tensor changes; the script verifies it.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEY = "model.language_model.layers.23.mlp.down_proj.weight"
WEIGHTS = "model.safetensors"
PAIRS = {"phase_boxed": ("v_boxed", "u_phase"), "pos_close": ("v_close", "u_pos"), "ovf_boxed": ("v_boxed_w", "u_ovf")}


def main():
    base, out_dir, edits = Path(sys.argv[1]), Path(sys.argv[3]), sys.argv[4:]
    if sys.argv[2].endswith(".json"):
        directions = {k: np.array(v, dtype=np.float32) for k, v in json.load(open(sys.argv[2])).items()}
    else:
        directions = np.load(sys.argv[2])
    with safe_open(base / WEIGHTS, "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        metadata = f.metadata()
    original = tensors[KEY].clone()
    weight = original.float()
    for edit in edits:
        name, delta = edit.split(":")
        v_name, u_name = PAIRS[name]
        v = torch.from_numpy(directions[v_name]).float()
        u = torch.from_numpy(directions[u_name]).float()
        weight += float(delta) * torch.outer(v, u)
    tensors[KEY] = weight.to(original.dtype)

    out_dir.mkdir(parents=True)
    save_file(tensors, out_dir / WEIGHTS, metadata=metadata or {"format": "pt"})
    for item in base.iterdir():
        if item.name not in (WEIGHTS, "delta-model.json"):
            shutil.copy(item, out_dir / item.name)
    with safe_open(out_dir / WEIGHTS, "pt") as f:
        changed = [k for k in f.keys() if not torch.equal(f.get_tensor(k), tensors[k] if k == KEY else tensors[k])]
        saved = f.get_tensor(KEY)
    if changed:
        raise SystemExit(f"round trip mismatch: {changed}")
    diff = (saved.float() - original.float())
    print(f"{out_dir.name}: {edits} changed {int((diff != 0).sum())} values, max {float(diff.abs().max()):.5f}, rel fro {float(diff.norm() / original.float().norm()):.5f}")


if __name__ == "__main__":
    main()
