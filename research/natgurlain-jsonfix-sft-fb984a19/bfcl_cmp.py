"""Single-turn tool calling (official harness shards under ../census/bfcl_NAME): per-category table for two models.
usage (from grpo/): python3 bfcl_cmp.py fc_s60 jf_s150"""
import glob
import json
import math
import sys


def cats(name):
    out = {}
    for f in glob.glob(f"../census/bfcl_{name}/*/result.json"):
        for c, v in json.load(open(f))["bfcl_single_turn"]["metrics"]["categories"].items():
            out[c] = (v["accuracy"], v["count"])
    return out


a_name, b_name = sys.argv[1], sys.argv[2]
A, B = cats(a_name), cats(b_name)
ta = tb = n = 0; var = 0.0
print(f"  category             n  {a_name:>7s}  {b_name:>7s}   diff")
for c in sorted(A):
    if c not in B:
        print(" ", c, "missing"); continue
    (a, na), (b, nb) = A[c], B[c]
    print(f"  {c:18s} {na:3d}  {100 * a:7.1f}  {100 * b:7.1f}  {100 * (b - a):+5.1f}")
    ta += a * na; tb += b * nb; n += na; var += a * (1 - a) * na + b * (1 - b) * nb
print(f"  all                {n:3d}  {100 * ta / max(1, n):7.1f}  {100 * tb / max(1, n):7.1f}  {100 * (tb - ta) / max(1, n):+5.1f}  (unpaired SE {100 * math.sqrt(var) / max(1, n):.1f})")
