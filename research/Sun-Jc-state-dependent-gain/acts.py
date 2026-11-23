"""Teacher-forced activation capture for stored completions.

Replays prompt + generated tokens through a checkpoint and stores hidden states at chosen
sites for a subsample of generated positions. No sampling: this is a deterministic forward.

Sites:  mlp_in:L   input of layer L's MLP (the vector a grafted neuron reads)
        premlp:L   residual stream just before layer L's MLP is added (after attention)
        resid:L    residual stream after layer L
        attn_o_in:L  per-head output of a full-attention layer (input of o_proj, 8 heads x 256)
        lin_o_in:L   per-head output of a linear-attention layer (input of out_proj, 16 heads x 128)

With mlp_in:23 and premlp:23 stored, any edit confined to the layer-23 MLP can be evaluated
offline: final hidden = premlp + MLP'(mlp_in), then the final norm and the output embedding.

Output: OUT/acts-K.pt with a dict
  X[site]   float16 [n_positions, 1024]
  meta      int32   [n_positions, 5] = (row index, step, steps since first </think> or -1, token id, total tokens)
  rows      list of {"id","sample","outcome","family","n","think"}

usage: python acts.py --model FP32_DIR --jobs JOBS --run RUN --out OUT --sites mlp_in:23,resid:15
                      [--every 8] [--shard K/P] [--max-rows N] [--only runaway,overflow,ok]
                      [--after-close K] [--from-step N]

The state stored for step s is the one that PRODUCED token s (its input is token s-1), so
with --after-close the state at step c is the one that emitted </think> and the states at
c+1..c+K are the first K states of the answer.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

from engine import THINK_CLOSE, Model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sites", default="mlp_in:23")
ap.add_argument("--every", type=int, default=8)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--only", default="")
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--after-close", type=int, default=0, help="also keep the K states following every </think>")
ap.add_argument("--from-step", type=int, default=0, help="keep only states at or after this step")
args = ap.parse_args()

k, p = (int(x) for x in args.shard.split("/"))
os.makedirs(args.out, exist_ok=True)
model = Model(args.model).install_fast_paths()
jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(args.run, "shard-*.jsonl"))) for l in open(f) if l.strip()]
rows.sort(key=lambda r: (r["id"], r["sample"]))
only = set(args.only.split(",")) if args.only else None


def outcome_of(r):
    closed = THINK_CLOSE in r["tokens"]
    return ("ok" if closed else "noclose") if r["finish"] == "stop" else ("overflow" if closed else "runaway")


rows = [r for r in rows if only is None or outcome_of(r) in only]
if args.max_rows:
    rows = rows[:args.max_rows]
rows = [r for i, r in enumerate(rows) if i % p == k]

sites = [(s.split(":")[0], int(s.split(":")[1])) for s in args.sites.split(",")]
captured = {}
hooks = []
for kind, layer in sites:
    name = f"{kind}:{layer}"
    block = model.text.layers[layer]
    if kind == "mlp_in":
        hooks.append(block.mlp.register_forward_pre_hook(lambda m, inp, name=name: captured.__setitem__(name, inp[0])))
    elif kind == "premlp":
        hooks.append(block.post_attention_layernorm.register_forward_pre_hook(
            lambda m, inp, name=name: captured.__setitem__(name, inp[0])))
    elif kind == "attn_o_in":  # per-head attention output (after the output gate), input of o_proj
        hooks.append(block.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, name=name: captured.__setitem__(name, inp[0])))
    elif kind == "lin_o_in":  # per-head linear-attention (DeltaNet) output, input of out_proj: 16 heads x 128
        hooks.append(block.linear_attn.out_proj.register_forward_pre_hook(
            lambda m, inp, name=name: captured.__setitem__(name, inp[0])))
    elif kind == "resid":
        hooks.append(block.register_forward_hook(
            lambda m, inp, out, name=name: captured.__setitem__(name, out[0] if isinstance(out, tuple) else out)))
    else:
        raise ValueError(kind)

X = {f"{kind}:{layer}": [] for kind, layer in sites}
meta, kept_rows = [], []
t0 = time.time()
with torch.inference_mode():
    for ri, r in enumerate(rows):
        prompt = model.encode_prompt(jobs[r["id"]]["prompt"])
        ids = torch.tensor([prompt + r["tokens"]], dtype=torch.long)
        n_prompt, n = len(prompt), len(r["tokens"])
        first_close = r["tokens"].index(THINK_CLOSE) if THINK_CLOSE in r["tokens"] else -1
        # Position j of the input predicts generated token j - n_prompt + 1; keep the state that
        # produced generated step `step` (input position n_prompt - 1 + step).
        steps = set(range(0, n, args.every))
        for c in (i for i, t in enumerate(r["tokens"]) if t == THINK_CLOSE):
            steps.update(range(c, min(n, c + args.after_close + 1)) if args.after_close else ())
        steps = sorted(s for s in steps if s >= args.from_step)
        want = torch.tensor([n_prompt - 1 + s for s in steps])
        past = model.new_cache(reserve=ids.shape[1] + 8)
        pieces = {name: [] for name in X}
        for a in range(0, ids.shape[1], args.chunk):
            b = min(ids.shape[1], a + args.chunk)
            model.text(input_ids=ids[:, a:b], past_key_values=past, use_cache=True)
            sel = want[(want >= a) & (want < b)] - a
            for name in X:
                pieces[name].append(captured[name][0, sel].to(torch.float16))
        for name in X:
            X[name].append(torch.cat(pieces[name]))
        for s in steps:
            since = -1 if first_close < 0 or s <= first_close else s - first_close
            meta.append((len(kept_rows), s, since, r["tokens"][s], n))
        kept_rows.append({"id": r["id"], "sample": r["sample"], "outcome": outcome_of(r), "family": jobs[r["id"]]["family"],
                          "n": n, "think": first_close if first_close >= 0 else n})
        if (ri + 1) % 10 == 0:
            print(f"[acts {k}/{p}] {ri + 1}/{len(rows)} rows, {(time.time() - t0) / 60:.1f} min", file=sys.stderr, flush=True)

torch.save({"X": {name: torch.cat(v) if v else torch.empty(0, 0, dtype=torch.float16) for name, v in X.items()},
            "meta": torch.tensor(meta, dtype=torch.int32), "rows": kept_rows}, os.path.join(args.out, f"acts-{k}.pt"))
print(f"[acts {k}/{p}] saved {len(meta)} positions from {len(kept_rows)} rows", file=sys.stderr, flush=True)
