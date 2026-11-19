"""Paired comparison of two run dirs (mean over seeds per item), bootstrap CI. usage: cmp.py BASE CAND"""
import json, sys, glob, random, collections, statistics as S
def load(d):
    per = collections.defaultdict(list); fam = {}
    for f in glob.glob(f"{d}/seed*.jsonl"):
        for l in open(f):
            r = json.loads(l); per[r["id"]].append(r["score"]); fam[r["id"]] = r["family"]
    return {k: S.mean(v) for k, v in per.items()}, fam
a, fam = load(sys.argv[1]); b, _ = load(sys.argv[2]); ids = sorted(set(a) & set(b))
d = [b[i] - a[i] for i in ids]; rng = random.Random(0)
bs = sorted(S.mean(rng.choices(d, k=len(d))) for _ in range(4000))
print(f"base {100*S.mean(a[i] for i in ids):.1f}  cand {100*S.mean(b[i] for i in ids):.1f}  delta {100*S.mean(d):+.2f}  95%CI [{100*bs[100]:+.2f},{100*bs[3900]:+.2f}]")
for f in ("gsm", "math", "arc"):
    fi = [i for i in ids if fam[i] == f]; print(f"  {f}: {100*S.mean(a[i] for i in fi):.1f} -> {100*S.mean(b[i] for i in fi):.1f}")
