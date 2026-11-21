"""Does writing a token's unembedding direction into the residual stream at an EARLY layer
still produce that token at the output?

For a few stored completions, replay the prefix up to a chosen step, then run the last
position once per configuration (layer, strength), adding strength * u to that layer's
output at that position only, where u is the unit vector along (1 + norm_weight) * E[token].
Reports the sampler-level probability of the token and what else moved.

usage: python inject_test.py --model FP32_DIR --jobs JOBS --run RUN --token 248069
                             [--layers 7,8,11,15,19,22,23] [--strengths 0.5,1,2,4] [--steps 6000,7500] [--rows 3]
"""
import argparse
import json

import torch

from engine import Model, Sampler
from scorelib import load_rows, outcome

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--token", type=int, default=248069)
ap.add_argument("--layers", default="7,8,11,15,19,22,23")
ap.add_argument("--strengths", default="0.5,1,2,4")
ap.add_argument("--steps", default="6000,7500")
ap.add_argument("--rows", type=int, default=3)
ap.add_argument("--only", default="runaway")
args = ap.parse_args()

model = Model(args.model).install_fast_paths()
sampler = Sampler()
jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
rows = [r for r in sorted(load_rows(args.run), key=lambda r: r["id"]) if outcome(r) in args.only.split(",")][:args.rows]
layers = [int(x) for x in args.layers.split(",")]
strengths = [float(x) for x in args.strengths.split(",")]
configs = [(None, 0.0)] + [(layer, s) for layer in layers for s in strengths]
K = len(configs)

E = model.text.embed_tokens.weight.detach()
gain = 1 + model.text.norm.weight.detach()
u = gain * E[args.token]
u = u / u.norm()
delta = {layer: torch.zeros(K, 1, 1024) for layer in layers}
for i, (layer, s) in enumerate(configs):
    if layer is not None:
        delta[layer][i, 0] = s * u


def make_hook(layer):
    def hook(module, inputs, output):
        if not active["on"]:
            return None
        if isinstance(output, tuple):
            return (output[0] + delta[layer],) + tuple(output[1:])
        return output + delta[layer]
    return hook


active = {"on": False}
for layer in layers:
    model.text.layers[layer].register_forward_hook(make_hook(layer))

print(f"token {args.token} {model.tok.decode([args.token])!r}; |(1+w)*E| = {float((gain * E[args.token]).norm()):.2f}; configs per state: {K}")
with torch.inference_mode():
    for r in rows:
        prompt = model.encode_prompt(jobs[r["id"]]["prompt"])
        for step in [int(x) for x in args.steps.split(",")]:
            if step >= r["n"]:
                continue
            seq = prompt + r["tokens"][:step]  # the state after these tokens predicts token `step`
            past, mask, _ = model.prefill([seq[:-1]])
            past.reorder_cache(torch.zeros(K, dtype=torch.long))
            mask = mask.expand(K, -1).contiguous()
            active["on"] = True
            _, _, h = model.step(torch.full((K,), seq[-1], dtype=torch.long), past, mask)
            active["on"] = False
            logits = model.head(h)
            seen = torch.zeros(K, model.vocab)
            seen[:, torch.tensor(sorted(set(r["tokens"][:step])))] = 1.0
            ids, q, vals = sampler.candidates(logits, seen)
            p_tok = (q * (ids == args.token)).sum(-1)
            z = logits - 1.5 * seen
            margin = z[:, args.token] - z.max(-1).values
            base_top = int(z[0].argmax())
            print(f"\n{r['id']} step {step}: baseline P = {float(p_tok[0]):.4f}, margin to top {float(margin[0]):+.1f}, top token {model.tok.decode([base_top])!r}, |h_final| rms {float(h[0].pow(2).mean().sqrt()):.2f}")
            for layer in layers:
                cells = []
                for i, (cl, s) in enumerate(configs):
                    if cl == layer:
                        top = int(z[i].argmax())
                        cells.append(f"s={s:g}: P={float(p_tok[i]):.3f} dlogit={float(logits[i, args.token] - logits[0, args.token]):+6.1f} top={model.tok.decode([top])!r:10s}")
                print(f"   layer {layer:2d}  " + " | ".join(cells))
