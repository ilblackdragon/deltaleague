"""Add the official score to generated rows, as the reward for a GRPO step.

usage: add_official.py PROMPTS.json ROWS.jsonl OUT.jsonl
"""
import json
import sys

from delta_cli.scoring import score_answer


def main():
    prompts = {row["id"]: row for row in json.load(open(sys.argv[1]))}
    with open(sys.argv[3], "w") as out:
        for line in open(sys.argv[2]):
            row = json.loads(line)
            prompt = prompts[row["id"]]
            if not prompt.get("scorer"):
                prompt = dict(prompt, scorer="choice" if prompt["kind"] == "letter" else "math")
            try:
                row["official"] = float(score_answer(prompt, row["text"]))
            except Exception:
                row["official"] = 0.0
            out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
