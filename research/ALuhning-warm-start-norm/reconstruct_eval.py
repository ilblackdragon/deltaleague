"""Rebuild the candidate checkpoint from a Delta submission with the CLI's own reconstruct(), so the evaluated
weights are exactly what the platform would rebuild. Copies config/tokenizer files from the parent dir.
Usage (Python with delta_cli): python reconstruct_eval.py PARENT_DIR SUBMISSION_DIR OUT_DIR"""
import json, os, shutil, sys, hashlib
from safetensors.torch import save_file
from safetensors import safe_open
from delta_cli.artifacts import reconstruct
parent, sub, out = sys.argv[1:4]
manifest = json.load(open(f"{sub}/manifest.json"))
result = reconstruct(f"{parent}/model.safetensors", f"{sub}/weights.safetensors", manifest,
                     allowed_prefix="model.language_model.")
with safe_open(f"{parent}/model.safetensors", "pt") as f:
    meta = f.metadata()
shutil.copytree(parent, out, ignore=shutil.ignore_patterns("model.safetensors"))
save_file({k: v.contiguous() for k, v in result.items()}, f"{out}/model.safetensors", metadata=meta)
print("reconstructed", len(manifest["tensors"]), "targets ->", out,
      hashlib.sha256(open(f"{out}/model.safetensors", "rb").read()).hexdigest())
