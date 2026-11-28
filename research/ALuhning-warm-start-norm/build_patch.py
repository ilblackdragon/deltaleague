"""Build an exact dense replacement patch and verify official reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from delta_cli.artifacts import reconstruct
from safetensors.torch import load_file, save_file


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def same(left, right):
    return torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    parent_file = args.parent / "model.safetensors"
    base = load_file(parent_file)
    candidate = load_file(args.candidate)
    if base.keys() != candidate.keys():
        raise ValueError("Checkpoint tensor keys differ")
    patch, mapping = {}, {}
    for key, before in sorted(base.items()):
        after = candidate[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Candidate tensor shape or dtype differs for {key}")
        if not same(before, after):
            if not key.startswith("model.language_model."):
                raise ValueError(f"Candidate changed a frozen tensor {key}")
            name = f"t{len(patch)}.values"
            patch[name] = after
            mapping[key] = {"values": name}
    args.out.mkdir()
    weights = args.out / "weights.safetensors"
    save_file(patch, weights)
    meta = json.loads((args.parent / "delta-model.json").read_text())
    manifest = {
        "schema_version": "delta.submission.v1", "reconstruction_version": "delta-reconstruct-v1",
        "model_config_id": meta["model_id"], "parent_version_id": meta["id"],
        "parent_content_hash": meta["content_hash"], "format": "dense-diff", "operation": "replace",
        "artifacts": [{"name": weights.name, "sha256": file_hash(weights), "size_bytes": weights.stat().st_size}],
        "tensors": mapping,
    }
    del base, patch
    rebuilt = reconstruct(parent_file, weights, manifest, allowed_prefix="model.language_model.")
    if rebuilt.keys() != candidate.keys() or not all(same(rebuilt[k], v) for k,v in candidate.items()):
        raise ValueError("Delta reconstruction differs from the evaluated checkpoint")
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidate_sha256": file_hash(args.candidate), "manifest_sha256": file_hash(manifest_path),
                      "weights_sha256": manifest["artifacts"][0]["sha256"], "size": weights.stat().st_size,
                      "modified_tensors": len(mapping), "reconstruction": "bitwise identical"}))


if __name__ == "__main__":
    main()
