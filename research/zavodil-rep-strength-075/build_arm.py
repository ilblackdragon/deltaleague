"""Build candidate checkpoints on the leader stack without any participant code.

usage: build_arm.py OUT_DIR BASE_DIR NO_REP_DIR RECIPE

BASE_DIR is the leader. NO_REP_DIR is the same stack without the repetition edit, so
leader - no_rep equals alpha=0.5 of the published L15/L19 edit, up to BF16 rounding.

Recipes:
  gain:F          multiply the final norm gain by F
  rep:A15:A19     set the repetition edit to strength A15 in layer 15 and A19 in layer 19
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"
NORM = "model.language_model.norm.weight"
REP = "model.language_model.layers.{}.mlp.down_proj.weight"
BASE_ALPHA = 0.5


def load_all(path):
    with safe_open(path, "pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}, handle.metadata()


def apply_gain(tensors, factor):
    # The final RMSNorm scales by (1 + weight), so the gain applies to that sum.
    norm = tensors[NORM]
    tensors[NORM] = ((1 + norm.float()) * factor - 1).to(norm.dtype)
    return [NORM]


def apply_rep(tensors, no_rep, alpha_by_layer):
    changed = []
    for layer, alpha in alpha_by_layer.items():
        key = REP.format(layer)
        without = no_rep[key].float()
        delta = tensors[key].float() - without
        tensors[key] = (without + delta * (alpha / BASE_ALPHA)).to(no_rep[key].dtype)
        changed.append(key)
    return changed


def main():
    out_dir, base_dir, no_rep_dir, recipe = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    tensors, metadata = load_all(base_dir / WEIGHTS)
    parent = {key: value.clone() for key, value in tensors.items()}
    kind, *args = recipe.split(":")

    if kind == "gain":
        expected = apply_gain(tensors, float(args[0]))
    elif kind == "rep":
        no_rep, _ = load_all(no_rep_dir / WEIGHTS)
        expected = apply_rep(tensors, no_rep, {15: float(args[0]), 19: float(args[1])})
    else:
        raise SystemExit(f"unknown recipe {recipe}")

    out_dir.mkdir(parents=True)
    save_file(tensors, out_dir / WEIGHTS, metadata=metadata or {"format": "pt"})
    for item in base_dir.iterdir():
        if item.name not in (WEIGHTS, "delta-model.json"):
            shutil.copy(item, out_dir / item.name)

    saved, _ = load_all(out_dir / WEIGHTS)
    changed = sorted(key for key in saved if not torch.equal(saved[key], parent[key]))
    if not set(changed) <= set(expected) or any(saved[key].dtype != parent[key].dtype for key in saved):
        raise SystemExit(f"unexpected changes: {changed}")
    counts = {key.split("language_model.")[-1]: int((saved[key] != parent[key]).sum()) for key in changed}
    print(f"{out_dir.name}: {recipe} changed {counts}")


if __name__ == "__main__":
    main()
