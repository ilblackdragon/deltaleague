"""Copy the lm-eval score from score_native.py output into generated rows, as the GRPO reward.

usage: merge_native.py ROWS.jsonl SCORES.json OUT.jsonl
"""
import json
import sys


def main():
    scores = {(str(r["id"]), r["sample"]): r["lm"] for r in json.load(open(sys.argv[2]))["rows"]}
    missing = 0
    with open(sys.argv[3], "w") as out:
        for line in open(sys.argv[1]):
            row = json.loads(line)
            key = (str(row["id"]), row["sample"])
            if key not in scores:
                missing += 1
                continue
            row["official"] = float(scores[key])
            out.write(json.dumps(row) + "\n")
    if missing:
        raise SystemExit(f"{missing} rows without a score")


if __name__ == "__main__":
    main()
