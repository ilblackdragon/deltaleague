"""Summarise decode-time oracle runs against the baseline run they replay.

For each oracle directory (one plan each): how many capped baseline completions were
replayed, how many now finish, how many score, and how the forced exit played out
(tokens written after the forced </think>, whether the completion hit the cap again).

Scores are paired with the baseline completion of the same prompt and sample, grouped by
how the baseline ended, and the net change is expressed in points on the whole pool.

usage: python oracle_report.py JOBS BASE_RUN ORACLE_DIR [ORACLE_DIR ...]
"""
import statistics
import sys

from scorelib import THINK_CLOSE, load_jobs, load_rows, outcome, score, tokenizer

jobs = load_jobs(sys.argv[1])
base_rows = load_rows(sys.argv[2])
tok = tokenizer()
text_of = lambda tokens: tok.decode(tokens, skip_special_tokens=True)

base = {(r["id"], r["sample"]): r for r in base_rows}
capped = {k: r for k, r in base.items() if outcome(r) in ("runaway", "overflow")}
score_cache = {}


def base_score(key):
    if key not in score_cache:
        score_cache[key] = score(jobs[key[0]], text_of(base[key]["tokens"]))[0]
    return score_cache[key]


n_all = len(base_rows)
capped_scoring = sum(base_score(k) for k in capped)
print(f"baseline: {n_all} completions, {len(capped)} capped "
      f"({sum(outcome(r) == 'runaway' for r in capped.values())} runaway, {sum(outcome(r) == 'overflow' for r in capped.values())} overflow), "
      f"capped completions scoring anyway: {capped_scoring:.0f}")

for d in sys.argv[3:]:
    rows = {(r["id"], r["sample"]): r for r in load_rows(d)}
    if not rows:
        print(f"{d}: no rows")
        continue
    plan = next(iter(rows.values()))["plan"]
    by_base, after = {}, []
    gained, lost = [], []
    for key, r in rows.items():
        s = score(jobs[key[0]], text_of(r["tokens"]))[0]
        cell = by_base.setdefault(r["base_outcome"], {"n": 0, "before": 0.0, "after": 0.0, "stop": 0})
        cell["n"] += 1
        cell["before"] += base_score(key)
        cell["after"] += s
        cell["stop"] += r["finish"] == "stop"
        (gained if s > base_score(key) else lost if s < base_score(key) else []).append(key)
        closes = [i for i, t in enumerate(r["tokens"]) if t == THINK_CLOSE]
        if closes and r["finish"] == "stop":
            after.append(r["n"] - closes[-1] - 1)
    med = f"{statistics.median(after):.0f}" if after else "-"
    print(f"\nplan {plan}  ({d.rstrip('/').split('/')[-1]}): replayed {len(rows)}; median answer length after the last close {med}")
    for name, c in sorted(by_base.items()):
        print(f"  base {name:8s}: n={c['n']:3d}  finished now {c['stop']:3d}  correct before {c['before']:3.0f} -> after {c['after']:3.0f}")
    delta = sum(c["after"] - c["before"] for c in by_base.values())
    print(f"  gained {len(gained)}, lost {len(lost)}  => {100 * delta / n_all:+.2f} points on this pool of {n_all}")
