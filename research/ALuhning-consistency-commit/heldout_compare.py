"""Paired held-out comparison (same problems, same seeds): original sample-0 vs candidate.
Usage: heldout_compare.py ITEMS BASE_PROBE CAND_PROBE"""
import json, random, sys
from grpo_gates import is_correct
g = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
def load(p, sample=None):
    out = {}
    for r in map(json.loads, open(p)):
        if sample is not None and r["sample"] != sample:
            continue
        t = r.get("text", ""); fin = t.split("</think>", 1)[1] if "</think>" in t else ""
        r["ok"] = int(r["finish_reason"] == "stop" and is_correct(r["bench"], fin, g[r["id"]]["answer"]))
        r["closed"] = int("</think>" in t); r["trunc"] = int(r["finish_reason"] == "length")
        out[r["id"]] = r
    return out
b = load(sys.argv[2], 0); c = load(sys.argv[3])
ids = sorted(b); assert all(b[i]["seed"] == c[i]["seed"] for i in ids)
for bench in ("gsm8k", "math", "all"):
    s = [i for i in ids if bench in ("all", b[i]["bench"])]
    tot = lambda d, k: sum(d[i][k] for i in s)
    d = [c[i]["ok"] - b[i]["ok"] for i in s]
    rng = random.Random(0); bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
    print("%-6s n=%d correct %d -> %d | closed %d -> %d | truncated %d -> %d | mean tokens %.0f -> %.0f | paired diff %+.3f [%+.3f, %+.3f] wins %d losses %d"
          % (bench, len(s), tot(b, "ok"), tot(c, "ok"), tot(b, "closed"), tot(c, "closed"), tot(b, "trunc"), tot(c, "trunc"),
             tot(b, "completion_tokens") / len(s), tot(c, "completion_tokens") / len(s), sum(d) / len(d), bs[250], bs[9750], d.count(1), d.count(-1)))
cw = sum(1 for i in ids if c[i]["closed"] and not c[i]["trunc"] and not c[i]["ok"])
print("candidate: closed and answered but wrong %d of %d closed" % (cw, sum(c[i]["closed"] for i in ids)))
