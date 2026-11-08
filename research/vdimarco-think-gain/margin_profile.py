"""Teacher-forced </think> margin profile over a long thinking trace.

Generates (or reuses) a long trace with the parent under server sampling settings, then
reports the distribution of logit(</think>) - max_logit across all positions, plus the
projection g(h) . mu_hat at those positions. Tells how large a bias would be needed to
fire at the trace's most "conclusive" moments.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, set_seed

THINK_END = 248069
model_dir, measure_json, out_path, max_new, prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
mu_hat = torch.tensor(json.loads(Path(measure_json).read_text())["summary"]["mu_hat"])
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(model_dir, config=cfg, local_files_only=True,
                                                   dtype=torch.float32, use_safetensors=True).eval()
captured = {}
model.model.language_model.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("g", o[0].detach().clone()))
rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
plen = ids.shape[1]
cache = Path(out_path).with_suffix(".trace.pt")
with torch.inference_mode():
    if cache.exists():
        out = torch.load(cache)
    else:
        set_seed(1000)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        torch.save(out, cache)
    logits = model(input_ids=out).logits[0].float()
g = captured["g"][plen - 1:-1]
lg = logits[plen - 1:-1]
margin = lg[:, THINK_END] - lg.max(-1).values
proj = g @ mu_hat
closed = THINK_END in out[0, plen:].tolist()
q = lambda t, p: t.quantile(p).item()
report = {"generated_tokens": int(out.shape[1] - plen), "closed_thinking": closed,
          "margin": {"min": margin.min().item(), "p50": q(margin, .5), "p90": q(margin, .9), "p99": q(margin, .99),
                     "max": margin.max().item(), "argmax_pos": int(margin.argmax())},
          "positions_within_3": int((margin > -3).sum()), "positions_within_6": int((margin > -6).sum()),
          "positions_within_10": int((margin > -10).sum()),
          "proj": {"min": proj.min().item(), "p50": q(proj, .5), "mean": proj.mean().item(), "max": proj.max().item()},
          "proj_at_top10_margin_positions": proj[margin.topk(10).indices].tolist(),
          "think_logit_at_top10_margin_positions": lg[margin.topk(10).indices, THINK_END].tolist(),
          "max_logit_at_top10_margin_positions": lg.max(-1).values[margin.topk(10).indices].tolist(),
          "think_logit_quantiles": {str(p): q(lg[:, THINK_END], p) for p in (.1, .5, .9, .99)},
          "scaled_row_fires": {str(sc): int(((sc * lg[:, THINK_END]) - lg.max(-1).values > -2).sum()) for sc in (1.2, 1.3, 1.4, 1.5)},
          "text_around_argmax": tok.decode(out[0, plen + max(0, int(margin.argmax()) - 40): plen + int(margin.argmax()) + 2])}
Path(out_path).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
