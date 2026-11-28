"""Paired comparison on the native-format measure, lm-eval metric, families weighted equally.

usage: nat_compare.py BASE CAND [CAND ...]   (names of results/native_NAME_seed1000)
"""
import json
import sys
from collections import defaultdict


def load(name):
    data = json.load(open(f"results/native_{name}_seed1000/native_scores.json"))
    tasks, bench, trunc = defaultdict(list), {}, []
    for row in data["rows"]:
        tasks[row["id"]].append(row["lm"])
        bench[row["id"]] = row["bench"]
        trunc.append(row.get("truncated", False))
    return tasks, bench, 100 * sum(trunc) / len(trunc)


def main():
    base, bench, base_trunc = load(sys.argv[1])
    print(f"base {sys.argv[1]} trunc {base_trunc:.1f}")
    for name in sys.argv[2:]:
        cand, _, trunc = load(name)
        per = defaultdict(list)
        for task, values in cand.items():
            if task in base:
                per[bench[task]].append(100 * (sum(values) / len(values) - sum(base[task]) / len(base[task])))
        parts = {k: sum(v) / len(v) for k, v in per.items()}
        variance = sum(sum((x - parts[k]) ** 2 for x in v) / (len(v) - 1) / len(v) for k, v in per.items())
        gain, se = sum(parts.values()) / len(parts), variance ** 0.5 / len(parts)
        print(f"{name}: gain {gain:+.2f} se {se:.2f} z {gain / se:+.2f} trunc {trunc:.1f} | " + " ".join(f"{k}:{v:+.1f}" for k, v in sorted(parts.items())))


if __name__ == "__main__":
    main()
