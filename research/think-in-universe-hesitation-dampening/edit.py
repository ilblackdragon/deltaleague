#!/usr/bin/env python3
"""Reproducible Delta League weight edit: hesitation-token damping + final-norm gain.

Reads the pristine parent checkpoint in ./model, writes ./edited.safetensors and
./edit-report.json. No randomness: the edit is a deterministic function of the
parent tensor values and the parameters below.

Parameters are part of the experiment record; change them only by committing a
new parameter block (and a new output file).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

MODEL_DIR = Path("model")
PARENT = MODEL_DIR / "model.safetensors"
OUT = Path("edited.safetensors")
REPORT = Path("edit-report.json")

EMBED = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"

# --- parameters -------------------------------------------------------------
EMBED_SCALE = 0.55          # multiplier for hesitation token rows (both directions: tied embeddings)
FINAL_NORM_GAIN = 1.35      # multiplier for the final RMSNorm gain (exact logit temperature scaling)
# Surface forms are tokenizer-dependent; every string below must be a single token.
HESITATION_TOKENS = [
    " Wait", "Wait", " wait", "wait",
    " Hmm", "Hmm", " hmm",
    " Actually", "Actually", " actually", "actually",
    " Alternatively", " Perhaps", "perhaps",
    " Maybe", "maybe",
]
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_ids = {}
    for text in HESITATION_TOKENS:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"{text!r} encodes to {ids}, not one token")
        token_ids[text] = ids[0]
    if len(set(token_ids.values())) != len(token_ids):
        raise SystemExit("duplicate token ids in hesitation set")

    tensors = {}
    with safe_open(PARENT, "pt") as f:
        keys = list(f.keys())
        for key in keys:
            tensors[key] = f.get_tensor(key)

    embed = tensors[EMBED]
    norm = tensors[FINAL_NORM]
    parent_dtype = embed.dtype
    before_embed_norms = embed[list(token_ids.values())].float().norm(dim=1)
    before_norm = norm.clone()

    # Deterministic, dtype-preserving transform computed in float32.
    embed_f = embed.float()
    for tid in token_ids.values():
        embed_f[tid] *= EMBED_SCALE
    tensors[EMBED] = embed_f.to(parent_dtype)

    norm_f = norm.float() * FINAL_NORM_GAIN
    tensors[FINAL_NORM] = norm_f.to(parent_dtype)

    # Parent must stay untouched; every other tensor must be bit-identical.
    changed = {EMBED, FINAL_NORM}
    with safe_open(PARENT, "pt") as f:
        for key in keys:
            if key in changed:
                continue
            if not torch.equal(f.get_tensor(key), tensors[key]):
                raise SystemExit(f"unintended change in {key}")

    save_file(tensors, str(OUT), metadata={"format": "pt"})

    report = {
        "parent_sha256": sha256(PARENT),
        "edited_sha256": sha256(OUT),
        "parent_size_bytes": PARENT.stat().st_size,
        "edited_size_bytes": OUT.stat().st_size,
        "parameters": {
            "embed_scale": EMBED_SCALE,
            "final_norm_gain": FINAL_NORM_GAIN,
            "hesitation_tokens": token_ids,
        },
        "changed_tensors": sorted(changed),
        "changed_values": {
            EMBED: len(token_ids) * embed.shape[1],
            FINAL_NORM: int(norm.numel()),
        },
        "hesitation_row_norms": {
            text: {"before": round(float(before_embed_norms[i]), 6),
                   "after": round(float(before_embed_norms[i] * EMBED_SCALE), 6)}
            for i, text in enumerate(token_ids)
        },
        "final_norm_mean": {"before": round(float(before_norm.float().mean()), 6),
                            "after": round(float(norm_f.mean()), 6)},
        "dtype": str(parent_dtype),
        "note": "Tied embeddings: scaling a row scales both its input embedding and its output logit coefficient. "
                "Scaling the final RMSNorm gain is exact logit temperature scaling (temperature x 1/1.35).",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("parent_sha256", "edited_sha256", "changed_values")}, indent=2))


if __name__ == "__main__":
    main()
