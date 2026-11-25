"""Pick the prompts that carry a learning signal from a mine.py output.

Kept: prompts where the samples disagree (some right, some wrong) or where any sample ran long.
Dropped: prompts the model always gets right quickly, or always gets wrong quickly.
"""
import json
import sys
from collections import Counter

src, dst = sys.argv[1], sys.argv[2]
long_at = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
rows = [json.loads(line) for line in open(src) if line.strip()]
kept, why = [], Counter()
for r in rows:
    mixed = 0 < sum(r["reward"]) < len(r["reward"])
    long_ = max(r["tokens"]) >= long_at
    if mixed or long_:
        kept.append(r["item"])
        why["mixed+long" if mixed and long_ else "mixed" if mixed else "long"] += 1
with open(dst, "w") as f:
    for it in kept:
        f.write(json.dumps(it) + "\n")
n = sum(len(r["reward"]) for r in rows)
print(json.dumps({
    "mined_prompts": len(rows), "kept": len(kept), "why": dict(why),
    "kept_by_pool": dict(Counter(it["pool"] for it in kept)), "mined_by_pool": dict(Counter(r["pool"] for r in rows)),
    "parent_score": round(100 * sum(sum(r["reward"]) for r in rows) / n, 2),
    "parent_trunc_8192": round(100 * sum(t >= 8192 for r in rows for t in r["tokens"]) / n, 2),
}))
