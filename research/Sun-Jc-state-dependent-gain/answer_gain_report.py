"""Report of the answer-phase gain experiment (answer_gain.py).

Every (completion, draw) is scored at each gain. Gains are compared with gain 1 on the same
completion and the same draw (common random numbers); the standard error treats prompts as the
independent units, since the draws of one prompt share its stored thinking.

usage: python answer_gain_report.py JOBS BASE_RUN OUT_DIR [--ifeval-detail]
"""
import collections
import glob
import json
import os
import sys

from scorelib import load_jobs, load_rows, score, tokenizer

jobs = load_jobs(sys.argv[1])
base = {(r["id"], r["sample"]): r for r in load_rows(sys.argv[2])}
out_dir = os.path.expanduser(sys.argv[3])
detail = "--ifeval-detail" in sys.argv
tok = tokenizer()

recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(out_dir, "shard-*.jsonl"))) for l in open(f) if l.strip()]
gains = sorted({r["gain"] for r in recs})
cell = {}  # (id, sample, draw) -> {gain: (score, finish, answer tokens, text)}
for r in recs:
    key = (r["id"], r["sample"])
    full = base[key]["tokens"][:r["close"] + 1] + r["answer"]
    text = tok.decode(full, skip_special_tokens=True)
    s = score(jobs[r["id"]], text)[0]
    cell.setdefault((r["id"], r["sample"], r["draw"]), {})[r["gain"]] = (s, r["finish"], len(r["answer"]), text)
complete = {k: v for k, v in cell.items() if all(g in v for g in gains)}
family = lambda k: jobs[k[0]]["family"]
print(f"{len(recs)} rows; {len(complete)} (completion, draw) cells with every gain; "
      f"{len({k[:2] for k in complete})} completions; gains {gains}")


def table(keys, label):
    prompts = sorted({k[:2] for k in keys})
    if not prompts:
        return
    print(f"\n{label}: {len(prompts)} completions, {len(keys)} cells")
    print(f"  {'gain':>5s} {'score':>7s} {'vs gain 1':>10s} {'+-se':>6s} {'gained':>7s} {'lost':>5s} {'capped':>7s} {'answer tokens (median)':>23s}")
    for g in gains:
        per_prompt = collections.defaultdict(list)
        gained = lost = capped = 0
        lengths = []
        for k in keys:
            d = complete[k][g][0] - complete[k][1.0][0]
            per_prompt[k[:2]].append((complete[k][g][0], d))
            gained += d > 0
            lost += d < 0
            capped += complete[k][g][1] == "length"
            lengths.append(complete[k][g][2])
        means = [sum(s for s, _ in v) / len(v) for v in per_prompt.values()]
        diffs = [sum(d for _, d in v) / len(v) for v in per_prompt.values()]
        n = len(diffs)
        mean_d = sum(diffs) / n
        se = (sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1) / n) ** 0.5
        lengths.sort()
        print(f"  {g:5.2f} {100 * sum(means) / n:7.1f} {100 * mean_d:+10.1f} {100 * se:6.1f} {gained:7d} {lost:5d} {capped:7d} {lengths[len(lengths) // 2]:23d}")


table(list(complete), "all families")
for fam in sorted({family(k) for k in complete}):
    table([k for k in complete if family(k) == fam], f"family {fam}")

if detail:
    from langdetect import DetectorFactory
    from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
    DetectorFactory.seed = 0
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for k, v in complete.items():
        job = jobs[k[0]]
        if job.get("scorer") != "ifeval":
            continue
        for g in gains:
            answer = v[g][3].split("</think>")[-1].strip()
            for iid, kw in zip(job["instruction_id_list"], job["kwargs"], strict=True):
                ins = INSTRUCTION_DICT[iid](iid)
                ins.build_description(**{a: b for a, b in kw.items() if b is not None})
                a = ins.get_instruction_args()
                if a and "prompt" in a:
                    ins.build_description(prompt=job["prompt"])
                c = per[iid][g]
                c[0] += 1
                c[1] += bool(answer) and bool(ins.check_following(answer))
    print("\nIFEval pass rate per instruction type (%), by gain:")
    print(f"  {'instruction':48s} {'n':>4s} " + " ".join(f"{g:6.2f}" for g in gains))
    for iid, by_gain in sorted(per.items(), key=lambda kv: kv[1][1.0][1] / kv[1][1.0][0]):
        n = by_gain[1.0][0]
        print(f"  {iid:48s} {n:4d} " + " ".join(f"{100 * by_gain[g][1] / by_gain[g][0]:6.1f}" for g in gains))
    total = {g: [sum(per[i][g][0] for i in per), sum(per[i][g][1] for i in per)] for g in gains}
    print(f"  {'all instructions':48s} {total[1.0][0]:4d} " + " ".join(f"{100 * total[g][1] / total[g][0]:6.1f}" for g in gains))
