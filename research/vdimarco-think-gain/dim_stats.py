"""Per-dimension statistics of the final-normed hidden state g(h) during thinking.

Looks for coordinates whose value is nearly constant across positions and prompts.
A constant coordinate d with value c lets an embedding edit E[t][d] += delta act as an
almost constant logit bias delta*c on token t.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, set_seed

model_dir, out_path, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = json.loads(Path(sys.argv[4]).read_text())
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(model_dir, config=cfg, local_files_only=True,
                                                   dtype=torch.float32, use_safetensors=True).eval()
captured = {}
model.model.language_model.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("g", o[0].detach().clone()))
G = []
with torch.inference_mode():
    for pi, prompt in enumerate(prompts):
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        set_seed(1000 + pi)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        # one teacher-forced pass over prompt + generation to capture g at every position
        model(input_ids=out)
        g = captured["g"]
        G.append(g[ids.shape[1] - 1:])  # positions that predict generated tokens
        print(f"prompt {pi}: {out.shape[1]-ids.shape[1]} tokens", file=sys.stderr)
G = torch.cat(G)
mean, std = G.mean(0), G.std(0)
ratio = mean.abs() / (std + 1e-6)
top = torch.topk(ratio, 12)
report = {"positions": G.shape[0], "dims": []}
for r, d in zip(top.values.tolist(), top.indices.tolist()):
    report["dims"].append({"dim": d, "mean": mean[d].item(), "std": std[d].item(), "abs_mean_over_std": r,
                           "min": G[:, d].min().item(), "max": G[:, d].max().item()})
E = model.model.language_model.embed_tokens.weight.detach()
report["e_think_at_top_dims"] = {str(d): E[248069, d].item() for d in top.indices.tolist()}
report["g_rms_mean"] = G.pow(2).mean(1).sqrt().mean().item()
Path(out_path).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
