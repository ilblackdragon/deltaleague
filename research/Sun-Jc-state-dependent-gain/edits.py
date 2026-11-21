"""Weight edits as data, applied identically to a live model and to a checkpoint file.

An edit set is a list of records (tensor name, kind, index, values):
  kind "row":  tensor[index, :] = values
  kind "col":  tensor[:, index] = values
  kind "all":  tensor[...]      = values
Tensor names are relative to `model.language_model.` (e.g. `layers.7.mlp.gate_proj.weight`).

Values are always rounded to bfloat16 first. The checkpoint is stored in bfloat16, so what
is evaluated locally is exactly what a submission built from the same edit set contains.
"""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.language_model."


def bf16(t):
    """Round to the nearest bfloat16 value, returned as float32."""
    return t.detach().to(torch.float32).to(torch.bfloat16).to(torch.float32)


def neuron(layer, index, gate, up, down):
    """Rewrite one SwiGLU neuron: out = down * silu(gate . x) * (up . x)."""
    base = f"layers.{layer}.mlp."
    return [(base + "gate_proj.weight", "row", index, bf16(gate)),
            (base + "up_proj.weight", "row", index, bf16(up)),
            (base + "down_proj.weight", "col", index, bf16(down))]


def _write(tensor, kind, index, values):
    values = values.to(tensor.dtype)
    if kind == "row":
        tensor[index] = values
    elif kind == "col":
        tensor[:, index] = values
    elif kind == "all":
        tensor.copy_(values)
    else:
        raise ValueError(kind)


def apply_to_model(model, edits):
    """Apply to an engine.Model in memory (float32 weights holding bfloat16 values)."""
    params = dict(model.text.named_parameters())
    cloned = set()
    with torch.no_grad():
        for name, kind, index, values in edits:
            p = params[name]
            if name not in cloned:  # weights are memory-mapped and shared between processes
                p.data = p.data.clone()
                cloned.add(name)
            _write(p.data, kind, index, bf16(values))
    return model


def apply_to_file(parent_file, out_file, edits):
    """Write a checkpoint equal to the parent except for the edited slices. Returns changed-element counts."""
    tensors = {}
    with safe_open(parent_file, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    changed = {}
    for name, kind, index, values in edits:
        key = PREFIX + name
        before = tensors[key].clone() if key not in changed else None
        if before is not None:
            changed[key] = before
        _write(tensors[key], kind, index, bf16(values))
    counts = {k: int((tensors[k] != before).sum()) for k, before in changed.items()}
    save_file(tensors, out_file, metadata=meta)
    return counts


def save(edits, path, note=""):
    torch.save({"edits": edits, "note": note}, path)


def load(path):
    return torch.load(path)["edits"]
