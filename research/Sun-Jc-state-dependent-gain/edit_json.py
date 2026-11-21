"""Exact, text-only form of an edit: every changed slice as bfloat16 bit patterns.

export:  python edit_json.py export EDITS.pt OUT.json [PARENT_ID=BUILT.safetensors ...]
apply:   python edit_json.py apply PARENT.safetensors EDIT.json OUT.safetensors
records: python edit_json.py records EDIT.json EDITS.pt
The JSON holds, per record, the tensor name (relative to model.language_model.), the kind
("row", "col" or "all"), the index and the values as unsigned 16-bit bfloat16 patterns. The
records replace whole rows, columns or tensors, so the same file applies to any parent that
shares the remaining values. `expected_sha256` holds, per parent version it was built on, the
SHA-256 of each edited tensor. `apply` writes the checkpoint and requires its edited tensors
to match one of those parents, which makes the result bit-identical to the one evaluated.
`records` writes the edit in the form the other scripts take with --edit (a local file).
"""
import hashlib
import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.language_model."
sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
bits = lambda t: (t.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).flatten().tolist()
unbits = lambda v: torch.tensor(v, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)

if sys.argv[1] == "export":
    blob = torch.load(sys.argv[2])
    out = {"note": blob["note"], "records": [], "expected_sha256": {}}
    for name, kind, index, values in blob["edits"]:
        out["records"].append({"tensor": name, "kind": kind, "index": index, "shape": list(values.shape), "bf16_bits": bits(values)})
    for built in sys.argv[4:]:  # the checkpoints built from these records, to record the resulting tensor hashes
        parent_id, path = built.split("=", 1)
        with safe_open(path, framework="pt") as f:
            out["expected_sha256"][parent_id] = {name: sha(f.get_tensor(PREFIX + name)) for name in sorted({r["tensor"] for r in out["records"]})}
    json.dump(out, open(sys.argv[3], "w"), separators=(",", ":"))
    print(f"{len(out['records'])} records, {sum(len(r['bf16_bits']) for r in out['records'])} values, "
          f"hashes for {len(out['expected_sha256'])} parent(s) -> {sys.argv[3]}")
elif sys.argv[1] == "records":
    spec = json.load(open(sys.argv[2]))
    edits = [(r["tensor"], r["kind"], r["index"], unbits(r["bf16_bits"]).float().reshape(r["shape"])) for r in spec["records"]]
    torch.save({"edits": edits, "note": spec["note"]}, sys.argv[3])
    print(f"{len(edits)} records -> {sys.argv[3]}")
else:
    parent, edit, out_path = sys.argv[2:5]
    spec = json.load(open(edit))
    tensors = {}
    with safe_open(parent, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for r in spec["records"]:
        t, v = tensors[PREFIX + r["tensor"]], unbits(r["bf16_bits"])
        assert t.dtype == torch.bfloat16
        if r["kind"] == "row":
            t[r["index"]] = v
        elif r["kind"] == "col":
            t[:, r["index"]] = v
        else:
            t.copy_(v.view_as(t))
    got = {name: sha(tensors[PREFIX + name]) for name in sorted({r["tensor"] for r in spec["records"]})}
    matches = [parent_id for parent_id, want in spec["expected_sha256"].items() if want == got]
    for name, digest in got.items():
        print(f"{name}: {digest}")
    assert matches, "the edited tensors match none of the recorded parents: wrong parent checkpoint?"
    print("edited tensors match the build on parent", matches[0])
    save_file(tensors, out_path, metadata=meta)
    print("file sha256:", hashlib.sha256(open(out_path, "rb").read()).hexdigest())
