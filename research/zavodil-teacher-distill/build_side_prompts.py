"""Build a side check set: code and instruction-following prompts, measured without execution.

usage: build_side_prompts.py OUT.json SEED humaneval=60 mbpp=60 ifeval=60

Rows carry kind "code" or "ifeval"; run_proxy scores them as never correct, but records
closure, truncation, token counts and the text, which is what the side check reads.
"""
import json
import random
import sys

from datasets import load_dataset


def main():
    out_path, seed = sys.argv[1], int(sys.argv[2])
    counts = {k: int(v) for k, v in (arg.split("=") for arg in sys.argv[3:])}
    rng = random.Random(seed)
    rows = []
    he = load_dataset("openai/openai_humaneval", split="test")
    for i in rng.sample(range(len(he)), counts["humaneval"]):
        rows.append({"bench": "humaneval", "kind": "code", "prompt": "Complete the following Python function.\n\n```python\n" + he[i]["prompt"] + "\n```", "answer": ""})
    mb = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for i in rng.sample(range(len(mb)), counts["mbpp"]):
        rows.append({"bench": "mbpp", "kind": "code", "prompt": mb[i]["prompt"] + "\nWrite the Python function. Its tests look like:\n" + "\n".join(mb[i]["test_list"][:1]), "answer": ""})
    ife = load_dataset("google/IFEval", split="train")
    for i in rng.sample(range(len(ife)), counts["ifeval"]):
        rows.append({"bench": "ifeval", "kind": "ifeval", "prompt": ife[i]["prompt"], "answer": ""})
    for index, row in enumerate(rows):
        row["id"] = 10000 + index
    json.dump(rows, open(out_path, "w"))
    print(len(rows), "side prompts ->", out_path)


if __name__ == "__main__":
    main()
