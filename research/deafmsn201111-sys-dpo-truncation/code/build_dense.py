"""Build a dense-diff (replace) submission: leader + LoRA v1 -> bf16 -> + DPO-b -> bf16, changed tensors only.
Then verify with delta_cli.artifacts.reconstruct that the platform rebuilds exactly these weights."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from delta_cli.artifacts import file_hash, reconstruct, validate_manifest
parent, out = Path(sys.argv[1]), Path(sys.argv[2]); loras = [Path(p) for p in sys.argv[3:]]
meta_p = json.loads((parent / "delta-model.json").read_text())
w = load_file(str(parent / "model.safetensors"))
changed = {}
for ld in loras:
    meta = json.loads((ld / "lora.json").read_text()); L = load_file(str(ld / "lora.safetensors"))
    s = meta["alpha"] / meta["rank"]
    for name in meta["targets"]:
        k = name.replace("model.layers.", "model.language_model.layers.", 1) + ".weight"
        W = changed.get(k, w[k]); changed[k] = (W.float() + s * (L[f"{name}.B"] @ L[f"{name}.A"])).to(W.dtype).contiguous()
out.mkdir(parents=True, exist_ok=True)
tensors, mapping = {}, {}
for i, k in enumerate(sorted(changed)):
    tensors[f"t{i}.values"] = changed[k]; mapping[k] = {"values": f"t{i}.values"}
path = out / "weights.safetensors"
save_file(tensors, str(path))
manifest = {"schema_version": "delta.submission.v1", "format": "dense-diff", "operation": "replace",
            "reconstruction_version": "delta-reconstruct-v1", "parent_version_id": meta_p["id"],
            "parent_content_hash": meta_p["content_hash"], "model_config_id": meta_p["model_id"], "tensors": mapping,
            "artifacts": [{"name": path.name, "sha256": file_hash(path), "size_bytes": path.stat().st_size}]}
validate_manifest(manifest)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
del tensors
rec = reconstruct(parent / "model.safetensors", path, manifest)
bad = [k for k in changed if not torch.equal(rec[k], changed[k])]
same = sum(torch.equal(rec[k], w[k]) for k in w if k not in changed)
print(f"tensors {len(mapping)}  size {path.stat().st_size/1e6:.0f} MB  mismatches {len(bad)}  untouched-equal {same}/{len(w)-len(changed)}")
