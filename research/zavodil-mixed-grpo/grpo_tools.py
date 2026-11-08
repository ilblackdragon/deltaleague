"""Helpers for the GRPO loop.

  grpo_tools.py init   BASE_DIR MASTER.safetensors   float32 master with the parent's tensor names
  grpo_tools.py export BASE_DIR MASTER.safetensors SERVE_DIR   BF16 checkpoint in the parent's exact layout
  grpo_tools.py sample POOL.json OUT.json COUNT SEED   random prompt subset, keeps stable ids
"""
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"


def init(base, master):
    with safe_open(Path(base) / WEIGHTS, "pt") as handle:
        tensors = {k: handle.get_tensor(k).float().contiguous() for k in handle.keys()}
    save_file(tensors, master)
    print(f"master {master}: {len(tensors)} tensors")


def export(base, master, serve):
    serve = Path(serve)
    serve.mkdir(parents=True, exist_ok=True)
    changed = 0
    tensors = {}
    with safe_open(Path(base) / WEIGHTS, "pt") as parent, safe_open(master, "pt") as trained:
        metadata = parent.metadata()
        for key in parent.keys():
            original = parent.get_tensor(key)
            value = trained.get_tensor(key).to(original.dtype)
            if not torch.equal(value, original):
                changed += 1
            tensors[key] = value.contiguous()
    save_file(tensors, serve / WEIGHTS, metadata=metadata or {"format": "pt"})
    for item in Path(base).iterdir():
        if item.name not in (WEIGHTS, "delta-model.json"):
            shutil.copy(item, serve / item.name)
    print(f"exported {serve}: {changed} tensors differ from parent")


def sample(pool, out, count, seed):
    rows = json.load(open(pool))
    picked = random.Random(int(seed)).sample(rows, int(count))
    json.dump(picked, open(out, "w"))
    print(f"{len(picked)} prompts -> {out}")


if __name__ == "__main__":
    {"init": init, "export": export, "sample": sample}[sys.argv[1]](*sys.argv[2:])
