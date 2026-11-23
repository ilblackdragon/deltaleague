"""Apply saved edit records to a BF16 checkpoint and report exactly what changed.

usage: python build_candidate.py PARENT.safetensors EDITS.pt OUT.safetensors
Prints, per edited tensor, the number of changed elements and the SHA-256 of the new tensor,
checks that every other tensor is bit-identical to the parent, and prints the file hash.
"""
import hashlib
import sys

import torch
from safetensors import safe_open

import edits as edit_lib

parent, edit_path, out = sys.argv[1:4]
records = edit_lib.load(edit_path)
print("edit:", torch.load(edit_path)["note"])
counts = edit_lib.apply_to_file(parent, out, records)
sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
edited_names = {edit_lib.PREFIX + name for name, *_ in records}
same = different = 0
with safe_open(parent, framework="pt") as a, safe_open(out, framework="pt") as b:
    assert set(a.keys()) == set(b.keys())
    for k in sorted(a.keys()):
        ta, tb = a.get_tensor(k), b.get_tensor(k)
        assert ta.dtype == tb.dtype and ta.shape == tb.shape, k
        if k in edited_names:
            print(f"  {k}: {tuple(tb.shape)} {tb.dtype}, {counts[k]} elements changed, sha256 {sha(tb)}")
            different += 1
        else:
            assert torch.equal(ta.view(torch.uint8) if not ta.is_floating_point() else ta.view(torch.int16) if ta.dtype == torch.bfloat16 else ta,
                               tb.view(torch.uint8) if not tb.is_floating_point() else tb.view(torch.int16) if tb.dtype == torch.bfloat16 else tb), k
            same += 1
    # the stored values must equal the edit records exactly
    for name, kind, index, values in records:
        t = b.get_tensor(edit_lib.PREFIX + name).float()
        got = t[index] if kind == "row" else t[:, index] if kind == "col" else t
        assert torch.equal(got, edit_lib.bf16(values)), name
print(f"{different} tensors edited, {same} tensors bit-identical to the parent, {sum(counts.values())} elements changed in total")
print("file sha256:", hashlib.sha256(open(out, "rb").read()).hexdigest())
