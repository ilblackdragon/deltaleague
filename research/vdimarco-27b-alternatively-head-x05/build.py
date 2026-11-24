"""Scale Alternatively-family output rows on the 27B Wait/Hmm leader.

Parent: 5065b8a8-e87d-4f06-b52b-183fcb3bab8c (mihaikoodo, PR #91 A + Wait/Hmm
output rows x0.25). This edit does not change those Wait/Hmm rows, the
</think> row, input embeddings, or the final RMSNorm. It only multiplies the
NVFP4 lm_head.weight_scale rows of three Alternatively tokens by 0.5, leaving
packed 4-bit codes untouched. x0.5 is exact in E4M3.

The builder reads only the selected rows from a local parent checkpoint
(CPU, a few kilobytes of tensor data). It does not materialize a full
edited.safetensors, which would need a second 21 GiB copy. The official
sparse-diff it writes is what `delta patch build` would produce for this
change. No network, no seeds, no credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

PARENT_VERSION = "5065b8a8-e87d-4f06-b52b-183fcb3bab8c"
PARENT_CONTENT_HASH = "d2574ed609c86067c8d42febe6f8869cea2615c1c2d5d820bde84e933f836b0b"
MODEL_ID = "qwen38-27b-nvfp4-ref"
SCALE_NAME = "lm_head.weight_scale"
PACKED_NAME = "lm_head.weight"
FACTOR = 0.5
# Discourse-marker tokens only. Bare "But" is not included: it is too common
# in ordinary answers. Extra "Hmm" spellings do not exist in this vocab.
TOKENS = {
    37201: "ĠAlternatively",
    66073: "Ġalternatively",
    88842: "Alternatively",
}
ALREADY_SCALED = {
    3655: "Ġwait",
    11158: "wait",
    13428: "ĠWait",
    13784: "Wait",
    77264: "Hmm",
    84485: "Ġhmm",
    85152: "ĠHmm",
    248069: "</think>",
}
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path):
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        raw = handle.read(header_len)
    header = json.loads(raw)
    data_start = 8 + header_len
    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
    return tensors, data_start, header.get("__metadata__")


def read_row(path: Path, data_start: int, spec: dict, row: int) -> bytes:
    rows, width = spec["shape"]
    if not 0 <= row < rows:
        raise ValueError(f"row {row} out of bounds for shape {spec['shape']}")
    bytes_per = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}[spec["dtype"]]
    row_bytes = width * bytes_per
    begin = data_start + spec["data_offsets"][0] + row * row_bytes
    with path.open("rb") as handle:
        handle.seek(begin)
        data = handle.read(row_bytes)
    if len(data) != row_bytes:
        raise ValueError(f"short read at row {row}")
    return data


def block_sq_norms(packed_row: torch.Tensor) -> torch.Tensor:
    lo, hi = packed_row & 0xF, packed_row >> 4
    vals = torch.stack([E2M1[lo.long()], E2M1[hi.long()]], -1).reshape(-1)
    return vals.reshape(-1, 16).pow(2).sum(-1)


def load_tokenizer_ids(tokenizer_path: Path) -> dict[str, int]:
    tokenizer = json.loads(tokenizer_path.read_text())
    lookup = dict(tokenizer["model"]["vocab"])
    lookup.update({item["content"]: item["id"] for item in tokenizer.get("added_tokens") or []})
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True, help="delta competition init model/ directory")
    parser.add_argument("--output", type=Path, required=True, help="submission directory")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = args.parent / "model.safetensors"
    meta = json.loads((args.parent / "delta-model.json").read_text())
    assert meta["id"] == PARENT_VERSION, meta["id"]
    assert meta["model_id"] == MODEL_ID
    assert meta["content_hash"] == PARENT_CONTENT_HASH
    lookup = load_tokenizer_ids(args.parent / "tokenizer.json")
    for token_id, text in {**TOKENS, **ALREADY_SCALED}.items():
        assert lookup[text] == token_id, (text, lookup.get(text), token_id)
    for text in TOKENS.values():
        assert text not in ALREADY_SCALED.values()

    specs, data_start, _ = read_header(model)
    scale_spec, packed_spec = specs[SCALE_NAME], specs[PACKED_NAME]
    assert scale_spec["dtype"] == "F8_E4M3" and scale_spec["shape"][1] == 320
    assert packed_spec["dtype"] == "U8" and packed_spec["shape"][1] == 2560
    assert scale_spec["shape"][0] == packed_spec["shape"][0] == 248320
    scale_index = sorted(specs).index(SCALE_NAME)
    assert scale_index == 2

    width = scale_spec["shape"][1]
    indices = []
    values = []
    rows = {}
    for token_id, text in sorted(TOKENS.items()):
        raw_scale = read_row(model, data_start, scale_spec, token_id)
        raw_packed = read_row(model, data_start, packed_spec, token_id)
        old = torch.frombuffer(bytearray(raw_scale), dtype=torch.float8_e4m3fn).clone()
        packed = torch.frombuffer(bytearray(raw_packed), dtype=torch.uint8).clone()
        new = (old.float() * FACTOR).clamp(max=448.0).to(torch.float8_e4m3fn)
        changed = new.view(torch.uint8) != old.view(torch.uint8)
        sq = block_sq_norms(packed)
        realized = math.sqrt(float((new.float() ** 2 * sq).sum()) / float((old.float() ** 2 * sq).sum()))
        nz = old.float() != 0
        per_block = new.float()[nz] / old.float()[nz]
        rows[text] = {
            "id": token_id,
            "target": FACTOR,
            "changed_blocks": int(changed.sum()),
            "lm_head_row_norm_factor": round(realized, 5),
            "lm_head_block_factor_min": round(float(per_block.min()), 5),
            "lm_head_block_factor_max": round(float(per_block.max()), 5),
        }
        assert torch.isfinite(new.float()).all()
        base = token_id * width
        for col in torch.nonzero(changed, as_tuple=False).flatten().tolist():
            indices.append(base + col)
            values.append(new[col])

    if not indices:
        raise SystemExit("empty_patch")
    index_tensor = torch.tensor(indices, dtype=torch.int64)
    value_tensor = torch.stack(values)
    assert (index_tensor[1:] > index_tensor[:-1]).all()

    ik, vk = f"t{scale_index}.indices", f"t{scale_index}.values"
    args.output.mkdir(parents=True, exist_ok=True)
    weights_path = args.output / "weights.safetensors"
    save_file({ik: index_tensor, vk: value_tensor}, str(weights_path))
    manifest = {
        "schema_version": "delta.submission.v1",
        "format": "sparse-diff",
        "operation": "replace",
        "reconstruction_version": "delta-reconstruct-v1",
        "parent_version_id": PARENT_VERSION,
        "parent_content_hash": PARENT_CONTENT_HASH,
        "model_config_id": MODEL_ID,
        "tensors": {SCALE_NAME: {"indices": ik, "values": vk}},
        "artifacts": [{
            "name": "weights.safetensors",
            "sha256": file_sha256(weights_path),
            "size_bytes": weights_path.stat().st_size,
        }],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "parent_version_id": PARENT_VERSION,
        "parent_content_hash": PARENT_CONTENT_HASH,
        "model_config_id": MODEL_ID,
        "evaluation": "qwen38-27b-nvfp4-agentic300-8k-v1",
        "factor": FACTOR,
        "tokens": rows,
        "changed_elements": len(indices),
        "tensor": SCALE_NAME,
        "sparse_tensor_index": scale_index,
        "weights_sha256": manifest["artifacts"][0]["sha256"],
        "weights_size_bytes": manifest["artifacts"][0]["size_bytes"],
        "seed": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
