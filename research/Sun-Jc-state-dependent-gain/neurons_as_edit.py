"""Express layer-23 MLP neurons of one checkpoint as an edit of another checkpoint.

Writes edit records (the form the other scripts take with --edit) holding the gate_proj and
up_proj rows and the down_proj columns of the given neurons, read from SOURCE. Applied to a
checkpoint that differs from SOURCE only in those neurons, the records turn it into SOURCE,
so verify_gain.py and edit_eval.py can measure what that difference does.

usage: python neurons_as_edit.py SOURCE/model.safetensors 1998,3529 OUT.pt
"""
import sys

from safetensors import safe_open

import edits as edit_lib

source, neurons, out = sys.argv[1], [int(i) for i in sys.argv[2].split(",")], sys.argv[3]
prefix = edit_lib.PREFIX + "layers.23.mlp."
records = []
with safe_open(source, framework="pt") as f:
    gate, up, down = (f.get_tensor(prefix + name + "_proj.weight").float() for name in ("gate", "up", "down"))
    for i in neurons:
        records += edit_lib.neuron(23, i, gate[i].clone(), up[i].clone(), down[:, i].clone())
edit_lib.save(records, out, note=f"layer-23 MLP neurons {neurons} of {source}")
print(f"{len(records)} records -> {out}")
