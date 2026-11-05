"""Build edited.safetensors: multiply the </think> embedding row by a gain s.

With tied embeddings and no logit bias, logit(</think>) = g(h) . E[</think>]. Scaling the row
by s scales that one logit by s. The change is conditional: when the model already leans
toward closing (raw logit around +15 to +20 while the top logit is around +26), the margin
closes by about (s-1)*16 nats; when the raw logit sits near zero (its median during rumination)
nothing changes; when negative it becomes slightly more negative. No other logit changes.

Input side: the first layer applies RMSNorm to the embedding, which removes a pure scale, so
the input-side effect is limited to the raw residual contribution of that one position.

Usage: edit_think_gain.py PARENT_DIR GAIN OUTPUT_SAFETENSORS
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

THINK_END = 248069
KEY = "model.language_model.embed_tokens.weight"
parent_dir, gain, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
tensors = load_file(str(Path(parent_dir) / "model.safetensors"))
E = tensors[KEY].clone()
assert E.dtype == torch.bfloat16
old = E[THINK_END].float()
new = (old * gain).to(torch.bfloat16)
E[THINK_END] = new
tensors[KEY] = E
save_file(tensors, out_path, metadata={"format": "pt"})
report = {"gain": gain, "old_norm": old.norm().item(), "new_norm": new.float().norm().item(),
          "entries_changed": int((new != old.to(torch.bfloat16)).sum()),
          "max_rel_rounding_error": float(((new.float() - old * gain).abs() / (old.abs() * gain + 1e-8)).max()),
          "output": out_path}
print(json.dumps(report, indent=1))
Path(out_path).with_suffix(".edit.json").write_text(json.dumps(report, indent=1))
