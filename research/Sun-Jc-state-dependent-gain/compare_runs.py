"""Paired comparison of two sampling runs on the same prompts (one completion per prompt and sample).

usage: python compare_runs.py JOBS RUN_A RUN_B [--label-a A --label-b B]
Prints, per family and overall: score, capped completions, thinking closed, mean tokens, and the
paired score difference B - A with its standard error (prompts as units) and gained/lost counts.
"""
import collections
import sys

from scorelib import load_jobs, load_rows, outcome, score, tokenizer

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
for name in ("--label-a", "--label-b"):
    if name in sys.argv:
        args.remove(sys.argv[sys.argv.index(name) + 1])
jobs = load_jobs(args[0])
tok = tokenizer()
la, lb = opt("--label-a", "A"), opt("--label-b", "B")


def scored(run):
    out = {}
    for r in load_rows(run):
        if r["id"] in jobs:
            out[(r["id"], r["sample"])] = (score(jobs[r["id"]], tok.decode(r["tokens"], skip_special_tokens=True))[0], outcome(r), r["n"])
    return out


a, b = scored(args[1]), scored(args[2])
keys = sorted(set(a) & set(b))
fam = lambda k: jobs[k[0]]["family"]
print(f"{len(keys)} prompts in both runs ({len(a)} in {la}, {len(b)} in {lb})")
print(f"{'family':8s} {'n':>4s} | {la + ' score':>12s} {'capped':>7s} {'tokens':>7s} | {lb + ' score':>12s} {'capped':>7s} {'tokens':>7s} | {'diff':>6s} {'+-se':>5s} {'gained':>6s} {'lost':>5s}")
for name in sorted({fam(k) for k in keys}) + ["all"]:
    ks = [k for k in keys if name == "all" or fam(k) == name]
    n = len(ks)
    d = [b[k][0] - a[k][0] for k in ks]
    mean = sum(d) / n
    se = (sum((x - mean) ** 2 for x in d) / max(n - 1, 1) / n) ** 0.5
    cap = lambda run: sum(run[k][1] in ("runaway", "overflow") for k in ks)
    print(f"{name:8s} {n:4d} | {100 * sum(a[k][0] for k in ks) / n:12.1f} {cap(a):7d} {sum(a[k][2] for k in ks) / n:7.0f} | "
          f"{100 * sum(b[k][0] for k in ks) / n:12.1f} {cap(b):7d} {sum(b[k][2] for k in ks) / n:7.0f} | {100 * mean:+6.1f} {100 * se:5.1f} {sum(x > 0 for x in d):6d} {sum(x < 0 for x in d):5d}")
moves = collections.Counter((a[k][1] in ("runaway", "overflow"), b[k][1] in ("runaway", "overflow")) for k in keys)
print(f"capped in both {moves[(True, True)]}, only in {la} {moves[(True, False)]}, only in {lb} {moves[(False, True)]}")
