"""Build a measurement set over the benchmark families the original proxy does not cover.

usage: build_wide.py OUT.json SEED PER_BENCH [FIRST_ID]

Uses the same public splits and dataset names as the public validation manifest. Rows carry the
official scorer name; generations are scored afterwards with score_official.py, which calls the
scoring module shared with the platform. Measurement only, never a training source.
"""
import json
import random
import sys

from datasets import load_dataset

LETTER = "\nThink it through, then give the letter of the correct option inside \\boxed{}."
EXACT = "\nThink it through, then reply with only the final answer and nothing else."
LETTERS = "ABCDE"


def choice_row(bench, question, options, gold_index):
    body = question + "\n\n" + "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(options))
    return {"bench": bench, "kind": "letter", "scorer": "choice", "prompt": body + LETTER, "answer": LETTERS[gold_index]}


def main():
    out, seed, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed)
    rows = []

    def pick(data):
        return [data[i] for i in rng.sample(range(len(data)), min(per, len(data)))]

    def lettered(row):
        labels = row["choices"]["label"]
        return row["choices"]["text"], labels.index(row["answerKey"])

    for row in pick(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")):
        texts, gold = lettered(row)
        rows.append(choice_row("arc_easy", row["question"], texts, gold))
    for row in pick(load_dataset("tau/commonsense_qa", split="validation")):
        texts, gold = lettered(row)
        rows.append(choice_row("commonsense_qa", row["question"], texts, gold))
    for row in pick(load_dataset("allenai/openbookqa", "main", split="test")):
        texts, gold = lettered(row)
        rows.append(choice_row("openbookqa", row["question_stem"], texts, gold))
    for row in pick(load_dataset("Rowan/hellaswag", split="validation")):
        rows.append(choice_row("hellaswag", "Choose the best continuation: " + row["ctx"], row["endings"], int(row["label"])))
    for row in pick(load_dataset("baber/piqa", split="validation")):
        rows.append(choice_row("piqa", row["goal"], [row["sol1"], row["sol2"]], int(row["label"])))
    for row in pick(load_dataset("allenai/winogrande", "winogrande_xl", split="validation")):
        rows.append(choice_row("winogrande", "Fill in the blank: " + row["sentence"], [row["option1"], row["option2"]], int(row["answer"]) - 1))
    for row in pick(load_dataset("aps/super_glue", "boolq", split="validation")):
        gold = 0 if int(row["label"]) == 1 else 1
        rows.append(choice_row("boolq", row["passage"] + "\n\nQuestion: " + row["question"] + "?", ["Yes", "No"], gold))
    for row in pick(load_dataset("cais/mmlu", "all", split="test")):
        rows.append(choice_row("mmlu", row["question"], row["choices"], int(row["answer"])))
    bbh = []
    for task in ("boolean_expressions", "date_understanding", "logical_deduction_three_objects", "object_counting", "web_of_lies", "sports_understanding"):
        bbh += list(load_dataset("SaylorTwift/bbh", task, split="test"))
    for row in pick(bbh):
        rows.append({"bench": "bbh", "kind": "code", "scorer": "exact", "prompt": row["input"] + EXACT, "answer": row["target"]})
    for row in pick(load_dataset("google/IFEval", split="train")):
        rows.append({"bench": "ifeval", "kind": "ifeval", "scorer": "ifeval", "prompt": row["prompt"], "answer": "",
                     "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]})
    for index, row in enumerate(rows):
        row["id"] = (int(sys.argv[4]) if len(sys.argv) > 4 else 20000) + index
    json.dump(rows, open(out, "w"))
    counts = {}
    for row in rows:
        counts[row["bench"]] = counts.get(row["bench"], 0) + 1
    print(len(rows), counts)


if __name__ == "__main__":
    main()
