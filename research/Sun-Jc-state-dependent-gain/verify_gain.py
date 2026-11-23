"""Check an edit against the real forward pass: effective logit gain per position.

Teacher-forces stored completions through the parent and through the edited model and, at
every generated position, regresses the edited logits on the parent's logits over the parent's
top-64 tokens. The slope is the effective gain; the residual sd says how much the logit
pattern changed beyond a pure gain. Also reports the total variation between the two
sampling distributions (evaluation sampler: presence penalty, top-k, top-p). The '>1.1' column
is the share of positions whose gain is above 1.1, i.e. switched toward a gain of 1.2.

usage: python verify_gain.py --model FP32_DIR --edit EDITS.pt --jobs JOBS --run RUN [--per-family 4] [--families a,b]
                             [--pool "NAME=fam,fam;NAME=fam"]
"""
import argparse
import collections
import json

import torch

import edits as edits_lib
from edit_eval import sampler_tv
from engine import THINK_CLOSE, Model, Sampler
from scorelib import load_rows, outcome

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--edit", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--per-family", type=int, default=4)
ap.add_argument("--families", default="")
ap.add_argument("--pool", default="", help="report groups of families as one population: NAME=fam,fam;NAME=fam")
ap.add_argument("--chunk", type=int, default=256)
args = ap.parse_args()

base = Model(args.model).install_fast_paths()
edited = Model(args.model).install_fast_paths()
edits_lib.apply_to_model(edited, edits_lib.load(args.edit))
sampler = Sampler()
jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
want = set(filter(None, args.families.split(",")))
pool = {fam: name for name, fams in (g.split("=") for g in filter(None, args.pool.split(";"))) for fam in fams.split(",")}
taken = collections.Counter()
rows = []
for r in sorted(load_rows(args.run), key=lambda r: (r["id"], r["sample"])):
    fam = jobs[r["id"]]["family"]
    if outcome(r) == "ok" and (not want or fam in want) and taken[fam] < args.per_family:
        taken[fam] += 1
        rows.append(r)

stats = collections.defaultdict(lambda: {"g": [], "res": [], "tv": []})
with torch.inference_mode():
    for r in rows:
        fam = jobs[r["id"]]["family"]
        prompt = base.encode_prompt(jobs[r["id"]]["prompt"])
        toks = r["tokens"]
        ids = torch.tensor([prompt + toks], dtype=torch.long)
        n_prompt, close = len(prompt), toks.index(THINK_CLOSE)
        first_seen = torch.full((base.vocab,), 1 << 30, dtype=torch.long)
        for i in range(len(toks) - 1, -1, -1):
            first_seen[toks[i]] = i
        past_b, past_e = base.new_cache(reserve=ids.shape[1] + 8), edited.new_cache(reserve=ids.shape[1] + 8)
        for a in range(0, ids.shape[1], args.chunk):
            b = min(ids.shape[1], a + args.chunk)
            hb = base.text(input_ids=ids[:, a:b], past_key_values=past_b, use_cache=True).last_hidden_state[0]
            he = edited.text(input_ids=ids[:, a:b], past_key_values=past_e, use_cache=True).last_hidden_state[0]
            lo = max(a, n_prompt - 1)
            if lo >= b:
                continue
            steps = torch.arange(lo - (n_prompt - 1), b - (n_prompt - 1))
            zb, ze = base.head(hb[lo - a:]), edited.head(he[lo - a:])
            top = zb.topk(64, dim=-1).indices
            xb, xe = zb.gather(1, top), ze.gather(1, top)
            xb, xe = xb - xb.mean(1, keepdim=True), xe - xe.mean(1, keepdim=True)
            slope = (xb * xe).sum(1) / xb.pow(2).sum(1)
            resid = (xe - slope[:, None] * xb).std(1)
            seen = (first_seen[None, :] < steps[:, None]).float()
            ids_b, q_b, _ = sampler.candidates(zb, seen)
            ids_e, q_e, _ = sampler.candidates(ze, seen)
            tv = sampler_tv(ids_b, q_b, ids_e, q_e)
            for phase, sel in (("thinking", steps <= close), ("answer", steps > close)):
                if int(sel.sum()):
                    s = stats[(pool.get(fam, fam), phase)]
                    s["g"].append(slope[sel]); s["res"].append(resid[sel]); s["tv"].append(tv[sel])
print(f"{len(rows)} completions ({dict(taken)})")
print(f"   {'family, phase':24s} {'n':>7s} {'gain mean':>10s} {'sd':>6s} {'q01':>6s} {'q50':>6s} {'q99':>6s} {'>1.1':>6s} {'>1.25':>6s} {'pattern sd':>11s} {'sampler TV mean':>16s} {'TV>0.2':>7s}")
for key in sorted(stats):
    g, res, tv = (torch.cat(stats[key][k]) for k in ("g", "res", "tv"))
    q = torch.quantile(g, torch.tensor([0.01, 0.5, 0.99]))
    print(f"   {key[0] + ', ' + key[1]:24s} {len(g):7d} {float(g.mean()):10.3f} {float(g.std()):6.3f} {float(q[0]):6.3f} {float(q[1]):6.3f} {float(q[2]):6.3f} "
          f"{float((g > 1.1).float().mean()):6.3f} {float((g > 1.25).float().mean()):6.3f} {float(res.mean()):11.4f} {float(tv.mean()):16.4f} {float((tv > 0.2).float().mean()):7.4f}")
