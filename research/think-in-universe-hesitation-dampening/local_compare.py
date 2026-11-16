#!/usr/bin/env python3
"""Paired comparison of two local_eval.py reports (same docs, same seeds)."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def exact_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    base = json.loads(open(args.baseline).read())
    cand = json.loads(open(args.candidate).read())
    for key in ("validation_sha256", "config_id", "model_id"):
        if base[key] != cand[key]:
            raise SystemExit(f"reports differ on {key}")
    b = {r["id"]: r for r in base["records"]}
    c = {r["id"]: r for r in cand["records"]}
    if set(b) != set(c):
        raise SystemExit("reports cover different examples")
    ids = sorted(b)
    diffs = np.array([c[i]["score"] - b[i]["score"] for i in ids], dtype=float)
    rng = np.random.default_rng(args.seed)
    boots = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean()
                      for _ in range(args.resamples)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    wins = int((diffs > 0).sum()); losses = int((diffs < 0).sum()); ties = int((diffs == 0).sum())
    result = {
        "n": len(ids),
        "baseline_mean": float(np.mean([b[i]["score"] for i in ids])),
        "candidate_mean": float(np.mean([c[i]["score"] for i in ids])),
        "difference": float(diffs.mean()),
        "difference_ci95": [float(lo), float(hi)],
        "wins": wins, "losses": losses, "ties": ties,
        "sign_test_p": exact_sign_test(wins, losses),
        "separable": bool(lo > 0 or hi < 0),
        "truncation": {
            "baseline": float(np.mean([b[i]["truncated"] for i in ids])),
            "candidate": float(np.mean([c[i]["truncated"] for i in ids])),
        },
        "mean_tokens": {
            "baseline": float(np.mean([b[i]["tokens"] for i in ids])),
            "candidate": float(np.mean([c[i]["tokens"] for i in ids])),
        },
        "hesitation_mentions_in_final_answer": {
            "baseline": float(np.mean([b[i]["hesitation_tokens"] for i in ids])),
            "candidate": float(np.mean([c[i]["hesitation_tokens"] for i in ids])),
        },
        "per_benchmark": {},
    }
    for name in sorted({b[i]["benchmark"] for i in ids}):
        sub = [i for i in ids if b[i]["benchmark"] == name]
        d = np.array([c[i]["score"] - b[i]["score"] for i in sub])
        result["per_benchmark"][name] = {
            "n": len(sub), "baseline": float(np.mean([b[i]["score"] for i in sub])),
            "candidate": float(np.mean([c[i]["score"] for i in sub])),
            "difference": float(d.mean()),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
        }
    text = json.dumps(result, indent=2)
    if args.output:
        open(args.output, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
