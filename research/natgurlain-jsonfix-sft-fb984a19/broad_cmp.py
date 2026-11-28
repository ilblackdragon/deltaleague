"""Paired comparison of two broad-evaluator runs (broad.sh): python broad_cmp.py ../broad/leader ../broad/fc_s60
Per benchmark: items, score A, score B, difference with a paired bootstrap interval; then all generative items pooled.
commonsense_qa is likelihood-scored (deterministic, no sampling) and is reported but kept out of the pooled figure."""
import glob
import json
import random
import sys
from collections import defaultdict


def load(root):
    items = {}
    for d in sorted(glob.glob(f"{root}/*/")):
        reps = glob.glob(d + "*.report.json")
        if not reps or not glob.glob(d + "result.json"):
            continue
        bench = d.rstrip("/").split("/")[-1].rsplit("_", 2)[0]
        rep = json.load(open(reps[0]))
        s = rep.get("samples")
        if isinstance(s, dict):
            for task, rows in s.items():
                for r in rows:
                    v = r.get("exact_match", r.get("acc"))
                    items[(bench, task, str(r["doc_id"]))] = float(v)
        elif isinstance(s, list):
            for r in s:
                items[(bench, "", str(r["id"]))] = 1.0 if r.get("passed") in (True, "True") else 0.0
    return items


def boot(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
by = defaultdict(list)
for k in keys:
    by[k[0]].append(k)
print(f"A {sys.argv[1]}: {len(a)} items; B {sys.argv[2]}: {len(b)} items; common {len(keys)}")
pooled = []
for bench, ks in by.items():
    d = [b[k] - a[k] for k in ks]
    lo, hi = boot(d)
    sa, sb = sum(a[k] for k in ks) / len(ks), sum(b[k] for k in ks) / len(ks)
    print(f"  {bench:16s} n={len(ks):5d}  A={100*sa:5.1f}  B={100*sb:5.1f}  diff={100*(sb-sa):+5.1f}  [{100*lo:+5.1f}, {100*hi:+5.1f}]")
    if bench != "commonsense_qa":
        pooled += d
if pooled:
    lo, hi = boot(pooled)
    print(f"  generative pooled n={len(pooled)}  diff={100*sum(pooled)/len(pooled):+.2f}  [{100*lo:+.2f}, {100*hi:+.2f}]")
