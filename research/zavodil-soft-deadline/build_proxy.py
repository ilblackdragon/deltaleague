"""Build a frozen prompt set from public datasets.

usage: build_proxy.py OUT.json SEED math=120 gsm=80 mmlu_pro=80 arc=40
"""
import json
import random
import sys

from datasets import load_dataset

BOXED = "\nThink it through, then give the final answer inside \\boxed{}."
LETTER = "\nThink it through, then give the letter of the correct option inside \\boxed{}."
LETTERS = "ABCDEFGHIJ"


def sample(dataset, count, rng):
    indices = rng.sample(range(len(dataset)), count)
    return [dataset[i] for i in indices]


def multiple_choice(question, labels, options):
    lines = [f"{label}. {option}" for label, option in zip(labels, options)]
    return question + "\n\n" + "\n".join(lines) + LETTER


def main():
    out_path, seed = sys.argv[1], int(sys.argv[2])
    counts = dict(arg.split("=") for arg in sys.argv[3:])
    rng = random.Random(seed)
    rows = []

    for row in sample(load_dataset("HuggingFaceH4/MATH-500", split="test"), int(counts["math"]), rng):
        rows.append({"bench": "math500", "kind": "math", "prompt": row["problem"] + BOXED, "answer": row["answer"]})

    for row in sample(load_dataset("openai/gsm8k", "main", split="test"), int(counts["gsm"]), rng):
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"bench": "gsm8k", "kind": "math", "prompt": row["question"] + BOXED, "answer": answer})

    for row in sample(load_dataset("TIGER-Lab/MMLU-Pro", split="test"), int(counts["mmlu_pro"]), rng):
        prompt = multiple_choice(row["question"], LETTERS, row["options"])
        rows.append({"bench": "mmlu_pro", "kind": "letter", "prompt": prompt, "answer": row["answer"]})

    for row in sample(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), int(counts["arc"]), rng):
        prompt = multiple_choice(row["question"], row["choices"]["label"], row["choices"]["text"])
        rows.append({"bench": "arc_challenge", "kind": "letter", "prompt": prompt, "answer": row["answerKey"]})

    for index, row in enumerate(rows):
        row["id"] = index
    with open(out_path, "w") as handle:
        json.dump(rows, handle)
    print(f"{len(rows)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
