"""Paired comparison of two mine.py outputs on the same prompt set."""
import json
import random
import sys
from collections import defaultdict


def load(path):
    return {r["key"]: r for r in (json.loads(line) for line in open(path) if line.strip())}


def summary(rows, cap=8192):
    r = [x for row in rows for x in row["reward"]]
    t = [x for row in rows for x in row["tokens"]]
    return {"prompts": len(rows), "samples": len(r), "score": round(100 * sum(r) / len(r), 2),
            "trunc": round(100 * sum(x >= cap for x in t) / len(t), 2), "mean_tokens": round(sum(t) / len(t))}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) & set(b))
    print("A", sys.argv[1], summary([a[k] for k in keys]))
    print("B", sys.argv[2], summary([b[k] for k in keys]))
    mean = lambda v: sum(v) / len(v)
    d = [mean(b[k]["reward"]) - mean(a[k]["reward"]) for k in keys]
    rng = random.Random(0)
    boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(4000))
    print(f"B - A = {100 * mean(d):+.2f} points, 95% bootstrap interval [{100 * boots[100]:+.2f}, {100 * boots[3899]:+.2f}], n={len(d)} prompts")
    by = defaultdict(list)
    for k, x in zip(keys, d):
        by[a[k]["pool"]].append(x)
    for pool, v in sorted(by.items()):
        se = (sum((x - mean(v)) ** 2 for x in v) / max(1, len(v) - 1) / len(v)) ** 0.5
        print(f"  {pool:8s} n={len(v):4d}  A={100 * mean([mean(a[k]['reward']) for k in keys if a[k]['pool'] == pool]):5.1f}  diff={100 * mean(v):+6.2f} ± {100 * se:.2f}")


if __name__ == "__main__":
    main()
