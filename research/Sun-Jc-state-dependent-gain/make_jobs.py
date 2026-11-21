"""Take the first N prompts of each family from pool.jsonl (pool order), as used for the 384-prompt runs.

usage: python make_jobs.py POOL.jsonl JOBS.jsonl [family=N ...]
default: gsm8k=80 math=120 arc=48 csqa=36 mmlu=36 ifeval=64
"""
import json
import sys

quota = {"gsm8k": 80, "math": 120, "arc": 48, "csqa": 36, "mmlu": 36, "ifeval": 64}
quota.update({k: int(v) for k, v in (a.split("=") for a in sys.argv[3:])})
taken = {k: 0 for k in quota}
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        item = json.loads(line)
        family = item["family"]
        if taken.get(family, 0) < quota.get(family, 0):
            taken[family] += 1
            out.write(json.dumps(item) + "\n")
print(taken)
