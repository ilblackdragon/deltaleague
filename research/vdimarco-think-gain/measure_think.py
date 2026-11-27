"""Measure the </think> logit margin and the mean final-norm direction during thinking.

Runs the parent on CPU for a few short prompts. Records, for each generated position
inside the thinking block:
  - logit of </think>, max logit, and the </think> rank
  - projection of the final-normed hidden state g(h) on the running mean direction
Writes JSON summary + per-step rows to the output path.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, set_seed

THINK_END = 248069
model_dir, out_path, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = json.loads(Path(sys.argv[4]).read_text())

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(model_dir, config=cfg, local_files_only=True,
                                                   dtype=torch.float32, use_safetensors=True).eval()
norm = model.model.language_model.norm
captured = {}
norm.register_forward_hook(lambda m, i, o: captured.__setitem__("g", o[:, -1, :].detach().clone()))

rows = []
gsum = torch.zeros(cfg.text_config.hidden_size)
for pi, prompt in enumerate(prompts):
    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
    set_seed(1000 + pi)
    past = None
    cur = ids
    generated = []
    t0 = time.monotonic()
    with torch.inference_mode():
        for step in range(max_new):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            g = captured["g"][0].float()
            # presence penalty 1.5 on generated tokens, as the server does
            pen = logits.clone()
            if generated:
                pen[torch.tensor(sorted(set(generated)))] -= 1.5
            probs = torch.softmax(pen / 1.0, -1)
            # top-k 20 / top-p 0.95 sampling
            topv, topi = torch.topk(probs, 20)
            cum = torch.cumsum(topv, 0)
            keep = cum - topv < 0.95
            topv, topi = topv[keep], topi[keep]
            nxt = topi[torch.multinomial(topv / topv.sum(), 1)].item()
            mx = logits.max().item()
            rank = int((logits > logits[THINK_END]).sum().item())
            rows.append({"prompt": pi, "step": step, "think_logit": logits[THINK_END].item(), "max_logit": mx,
                         "margin": logits[THINK_END].item() - mx, "rank": rank,
                         "p_think_penalized": probs[THINK_END].item(), "g_norm": g.norm().item(),
                         "g": g.tolist() if step % 8 == 0 else None, "token": nxt})
            gsum += g
            generated.append(nxt)
            cur = torch.tensor([[nxt]])
            if nxt == THINK_END or nxt in (248044, 248046):
                break
    text = tok.decode(generated, skip_special_tokens=False)
    print(f"prompt {pi}: {len(generated)} tokens in {time.monotonic()-t0:.0f}s; ended={'</think>' in text}", file=sys.stderr)
    print(text[-400:].replace("\n", " "), file=sys.stderr)

mu = gsum / len(rows)
mu_hat = mu / mu.norm()
# projection of every captured g on mu_hat
proj = [torch.tensor(r["g"]) @ mu_hat for r in rows if r["g"] is not None]
proj = torch.stack(proj)
E = model.model.language_model.embed_tokens.weight.detach()
e_think = E[THINK_END].float()
summary = {
    "steps": len(rows), "prompts": len(prompts),
    "margin_median": float(torch.tensor([r["margin"] for r in rows]).median()),
    "margin_p90": float(torch.tensor([r["margin"] for r in rows]).quantile(0.9)),
    "rank_median": float(torch.tensor([float(r["rank"]) for r in rows]).median()),
    "g_norm_mean": float(torch.tensor([r["g_norm"] for r in rows]).mean()),
    "proj_mean": proj.mean().item(), "proj_std": proj.std().item(), "proj_min": proj.min().item(), "proj_max": proj.max().item(),
    "mu_norm": mu.norm().item(), "e_think_norm": e_think.norm().item(),
    "cos_e_think_mu": float(e_think @ mu_hat / e_think.norm()),
    "mu_hat": mu_hat.tolist(),
}
for r in rows:
    r.pop("g")
Path(out_path).write_text(json.dumps({"summary": summary, "rows": rows}))
print(json.dumps({k: v for k, v in summary.items() if k != "mu_hat"}, indent=1))
