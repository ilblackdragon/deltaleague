"""Build a prompt set from train splits only, for generating traces to train on.

usage: build_train_prompts.py OUT.json SEED math=500 gsm=400 arc=300

Test splits are never loaded here: they stay reserved for measurement.
"""
import json
import random
import sys

from datasets import load_dataset

from build_proxy import BOXED, multiple_choice

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def boxed_answer(solution):
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    depth, index = 0, start + len("\\boxed")
    for position in range(index, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[position], 0)
        if depth == 0:
            return solution[index + 1 : position]
    return None


def main():
    out_path, seed = sys.argv[1], int(sys.argv[2])
    counts = {key: int(value) for key, value in (arg.split("=") for arg in sys.argv[3:])}
    rng = random.Random(seed)
    rows = []

    math_rows = []
    for config in MATH_CONFIGS:
        for row in load_dataset("EleutherAI/hendrycks_math", config, split="train"):
            answer = boxed_answer(row["solution"])
            if answer:
                math_rows.append({"bench": "math_train", "kind": "math", "prompt": row["problem"] + BOXED, "answer": answer})
    rows += rng.sample(math_rows, counts["math"])

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for index in rng.sample(range(len(gsm)), counts["gsm"]):
        answer = gsm[index]["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"bench": "gsm_train", "kind": "math", "prompt": gsm[index]["question"] + BOXED, "answer": answer})

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    for index in rng.sample(range(len(arc)), counts["arc"]):
        row = arc[index]
        prompt = multiple_choice(row["question"], row["choices"]["label"], row["choices"]["text"])
        rows.append({"bench": "arc_train", "kind": "letter", "prompt": prompt, "answer": row["answerKey"]})

    for index, row in enumerate(rows):
        row["id"] = index
    with open(out_path, "w") as handle:
        json.dump(rows, handle)
    print(f"{len(rows)} train prompts -> {out_path}")


if __name__ == "__main__":
    main()
