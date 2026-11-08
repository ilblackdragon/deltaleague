"""Step 3b: merge a LoRA into the parent's BF16 weights -> a full checkpoint dir for vLLM / delta patch build.

W' = W + dose * (alpha / r) * B @ A, computed in float32 and cast back to the parent dtype,
which matches Delta's LoRA reconstruction rule. All other tensors are copied unchanged.
HF text-model names (model.layers.N...) map to checkpoint names (model.language_model.layers.N...).

Usage: python merge_lora.py PARENT_DIR LORA_DIR OUT_DIR [--dose 1.0]
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("parent")
ap.add_argument("lora")
ap.add_argument("out")
ap.add_argument("--dose", type=float, default=1.0)
args = ap.parse_args()

parent, lora_dir, out = Path(args.parent), Path(args.lora), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((lora_dir / "lora.json").read_text())
lora = load_file(str(lora_dir / "lora.safetensors"))
scale = args.dose * meta["alpha"] / meta["rank"]

shards = sorted(parent.glob("*.safetensors"))
assert len(shards) == 1, f"expected one weights file, got {shards}"
src = shards[0]
with safe_open(str(src), "pt") as f:
    st_meta = f.metadata()
weights = load_file(str(src))

changed = 0
for name in meta["targets"]:
    key = name.replace("model.layers.", "model.language_model.layers.", 1) + ".weight"
    assert key in weights, f"missing {key}"
    W = weights[key]
    A, B = lora[f"{name}.A"], lora[f"{name}.B"]
    assert W.shape == (B.shape[0], A.shape[1]), (key, W.shape, A.shape, B.shape)
    weights[key] = (W.float() + scale * (B @ A)).to(W.dtype).contiguous()
    changed += 1

save_file(weights, str(out / "model.safetensors"), metadata=st_meta)
for p in parent.iterdir():
    if p.is_file() and p.suffix != ".safetensors" and p.name not in ("model.safetensors.index.json",):
        shutil.copy2(p, out / p.name)
print(f"merged {changed} tensors at dose {args.dose} (scale {scale:.3f}) -> {out / 'model.safetensors'}")
