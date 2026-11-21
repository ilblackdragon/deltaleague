"""Build local prompt sets from public TRAIN splits in the platform's prompt format.

Every math and multiple-choice prompt ends with the platform's own instruction line
(benchmarks/heldout-v1/workflow-pass2.js). Train splits are disjoint from the public
validation selection, which draws on test splits. IFEval has a single split; its items are
used for measurement only.

usage: python data.py OUT_DIR
Writes OUT_DIR/pool.jsonl with {"id","family","prompt","answer","scorer",...} per line.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
out_dir = Path(sys.argv[1]).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
rng = random.Random(20261113)
pool = []


def add(family, raw_id, prompt, answer, scorer, **extra):
    uid = f"{family}:{hashlib.sha256(str(raw_id).encode()).hexdigest()[:10]}"
    pool.append({"id": uid, "family": family, "prompt": prompt, "answer": answer, "scorer": scorer, **extra})


def boxed(solution):
    """Last \\boxed{...} content of a MATH solution, with balanced braces."""
    i = solution.rfind("\\boxed")
    if i < 0:
        return None
    j = solution.find("{", i)
    depth, start = 0, j
    for pos in range(j, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[pos], 0)
        if depth == 0:
            return solution[start + 1:pos]
    return None


# Grade-school math (GSM8K train).
gsm = load_dataset("openai/gsm8k", "main", split="train")
for i in rng.sample(range(len(gsm)), 400):
    row = gsm[i]
    add("gsm8k", f"gsm-train-{i}", row["question"].strip() + SUFFIX, row["answer"].split("####")[-1].strip().replace(",", ""), "math")

# Competition math (MATH train, all subjects), levels 2-5.
math_rows = []
for subject in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
    ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
    for i, row in enumerate(ds):
        level = int(re.sub(r"\D", "", row["level"]) or 0)
        ans = boxed(row["solution"])
        if ans and 2 <= level <= 5:
            math_rows.append((f"math-train-{subject}-{i}", row["problem"].strip(), ans, level, subject))
for raw_id, problem, ans, level, subject in rng.sample(math_rows, 500):
    add("math", raw_id, problem + SUFFIX, ans, "math", level=level, subject=subject)


def choice_prompt(stem, options):
    letters = "ABCDE"[:len(options)]
    return stem.strip() + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options)) + SUFFIX


# Science multiple choice (ARC-Challenge train), four lettered options only.
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
arc_rows = [r for r in arc if r["choices"]["label"] == ["A", "B", "C", "D"]]
for row in rng.sample(arc_rows, 200):
    add("arc", f"arc-train-{row['id']}", choice_prompt(row["question"], row["choices"]["text"]), row["answerKey"], "choice")

# Commonsense multiple choice (CommonsenseQA train), five options.
csqa = load_dataset("tau/commonsense_qa", split="train")
for i in rng.sample(range(len(csqa)), 150):
    row = csqa[i]
    add("csqa", f"csqa-train-{row['id']}", choice_prompt(row["question"], row["choices"]["text"]), row["answerKey"], "choice")

# Academic multiple choice (MMLU auxiliary train).
mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
for i in rng.sample(range(len(mmlu)), 150):
    row = mmlu[i]
    add("mmlu", f"mmlu-aux-{i}", choice_prompt(row["question"], row["choices"]), "ABCD"[row["answer"]], "choice")

# Instruction following (Google IFEval; single split).
ife = load_dataset("google/IFEval", split="train")
for i in rng.sample(range(len(ife)), 200):
    row = ife[i]
    add("ifeval", f"ifeval-{row['key']}", row["prompt"], None, "ifeval",
        instruction_id_list=row["instruction_id_list"], kwargs=row["kwargs"])

with open(out_dir / "pool.jsonl", "w") as f:
    for item in pool:
        f.write(json.dumps(item) + "\n")
counts = {}
for item in pool:
    counts[item["family"]] = counts.get(item["family"], 0) + 1
print("pool:", counts, "->", out_dir / "pool.jsonl")
