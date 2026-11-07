"""Paired comparison on the official scorer over proxy320 and wide270, benchmarks weighted equally.

usage: compare_official.py BASE:seed,seed CAND:seed,seed [...]
Scores are cached in results/<run>/official.json. A run NAME uses results/NAME_seedS and results/wide_NAME_seedS.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from delta_cli.scoring import score_answer

PROMPTS = {}
for path in ("proxy/proxy320.json", "proxy/wide270.json", "proxy/wideB.json"):
    for row in json.load(open(path)):
        PROMPTS[row["id"]] = row


def example(prompt):
    if prompt.get("scorer"):
        return prompt
    return dict(prompt, scorer="choice" if prompt["kind"] == "letter" else "math")


def scored(run):
    folder = Path("results") / run
    cache = folder / "official.json"
    if cache.exists():
        return json.load(open(cache))
    rows = []
    for line in open(folder / "rows.jsonl"):
        row = json.loads(line)
        try:
            value = score_answer(example(PROMPTS[row["id"]]), row["text"])
        except Exception:
            value = 0.0
        rows.append([row["id"], row["bench"], value, row["truncated"], row["tokens"]])
    json.dump(rows, open(cache, "w"))
    return rows


def load(spec):
    name, seeds = spec.split(":")
    tasks, benches, trunc, total = defaultdict(list), {}, 0, 0
    for seed in seeds.split(","):
        for run in (f"{name}_seed{seed}", f"wide_{name}_seed{seed}", f"wideB_{name}_seed{seed}"):
            if not (Path("results") / run / "rows.jsonl").exists():
                continue
            for task, bench, value, truncated, _ in scored(run):
                tasks[task].append(value)
                benches[task] = bench
                trunc += truncated
                total += 1
    return name, tasks, benches, 100 * trunc / max(total, 1)


def main():
    base_name, base, benches, base_trunc = load(sys.argv[1])
    print(f"base {base_name} trunc {base_trunc:.1f}")
    for spec in sys.argv[2:]:
        name, cand, _, trunc = load(spec)
        per = defaultdict(list)
        for task in cand:
            if task in base:
                per[benches[task]].append(100 * (sum(cand[task]) / len(cand[task]) - sum(base[task]) / len(base[task])))
        parts = {}
        variance = 0.0
        for bench, diffs in per.items():
            mean = sum(diffs) / len(diffs)
            parts[bench] = round(mean, 1)
            variance += sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)
        count = len(per)
        gain = sum(parts.values()) / count
        se = variance ** 0.5 / count
        print(f"{name}: benches {count} gain {gain:+.2f} se {se:.2f} z {gain / se if se else 0:+.2f} trunc {trunc:.1f}")
        print("   " + " ".join(f"{k}:{v:+.0f}" for k, v in sorted(parts.items())))


if __name__ == "__main__":
    main()
