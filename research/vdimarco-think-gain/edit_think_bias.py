"""Build edited.safetensors: push the </think> embedding along the mean final-norm direction.

Qwen3.5-0.8B ties input and output embeddings and has no logit bias. The logit of token t
is g(h) . E[t], where g(h) is the final RMSNorm output. During thinking, g(h) has a strong
shared component mu (projection on the unit mean direction is positive at every measured
position). Adding delta * mu_hat to E[</think>] therefore raises the </think> logit by
delta * (g(h) . mu_hat) > 0 at every position, which acts as an approximately constant
bias toward closing the thinking block. No other token's logit changes.

Usage: edit_think_bias.py PARENT_DIR MEASURE_JSON TARGET_BIAS OUTPUT_SAFETENSORS
TARGET_BIAS is the intended median logit shift in nats; delta = TARGET_BIAS / median(proj).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

THINK_END = 248069
KEY = "model.language_model.embed_tokens.weight"
parent_dir, measure_path, target, out_path = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
summary = json.loads(Path(measure_path).read_text())["summary"]
mu_hat = torch.tensor(summary["mu_hat"], dtype=torch.float32)
assert abs(mu_hat.norm().item() - 1) < 1e-4
delta = target / summary["proj_mean"]

tensors = load_file(str(Path(parent_dir) / "model.safetensors"))
E = tensors[KEY]
assert E.dtype == torch.bfloat16 and E.shape[0] > THINK_END
old = E[THINK_END].float()
new = (old + delta * mu_hat).to(torch.bfloat16)
E = E.clone()
E[THINK_END] = new
tensors[KEY] = E
changed = int((new != old.to(torch.bfloat16)).sum())
save_file(tensors, out_path, metadata={"format": "pt"})
report = {"delta": delta, "target_median_bias_nats": target, "proj_mean": summary["proj_mean"],
          "proj_min": summary["proj_min"], "proj_max": summary["proj_max"],
          "implied_shift_min": delta * summary["proj_min"], "implied_shift_max": delta * summary["proj_max"],
          "old_norm": old.norm().item(), "new_norm": new.float().norm().item(),
          "cos_old_new": float(old @ new.float() / (old.norm() * new.float().norm())),
          "entries_changed": changed, "output": out_path}
print(json.dumps(report, indent=1))
Path(out_path).with_suffix(".edit.json").write_text(json.dumps(report, indent=1))
