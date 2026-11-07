"""Rebuild little Qwen v0.1 from a Delta League parent checkpoint. Two edits, nothing fit to evaluation data:
  1. layer 16 MLP neuron 998's output column x -2 (the neuron that writes "Wait" while thinking)
  2. the layer-13 doubt direction d projected out of everything layers 0-13 write to the residual stream:
     W <- W - d (d . W) for each o_proj / out_proj / down_proj in layers 0-13 (28 tensors)
Usage: python build_edit.py PARENT/model.safetensors doubt_direction_L13.json edited.safetensors
Needs torch and safetensors. Prints the sha256 of every changed tensor."""
import sys, json, hashlib, torch
from safetensors.torch import load_file, save_file

parent, direction, out = sys.argv[1:4]
w = load_file(parent)
d = torch.tensor(json.load(open(direction))["values"], dtype=torch.float32)
d = d / d.norm()
changed = []
for layer in range(14):
    for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
        k = f"model.language_model.layers.{layer}.{part}.weight"
        if k in w:
            W = w[k].float()
            W -= torch.outer(d, d @ W)
            w[k] = W.to(w[k].dtype)
            changed.append(k)
k = "model.language_model.layers.16.mlp.down_proj.weight"
w[k][:, 998] *= -2
changed.append(k)
save_file(w, out, metadata={"format": "pt"})
for k in sorted(changed):
    t = w[k].contiguous()
    print(hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest(), k)
print(len(changed), "tensors changed")
