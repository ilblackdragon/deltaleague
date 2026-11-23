"""A small end-to-end check set made of pool prompts that the 384-prompt job file did not use.

Per reasoning family, the prompts that follow the job file's in pool order (as many as the job
file took), thinned to an evenly spaced subset; for IFEval, an evenly spaced subset of all
pool prompts outside the job file.

usage: python make_e2e_jobs.py POOL.jsonl JOBS.jsonl OUT.jsonl
default subset: math=24 gsm8k=16 arc=10 csqa=7 mmlu=7 ifeval=32
"""
import collections
import json
import sys

SUBSET = {"math": 24, "gsm8k": 16, "arc": 10, "csqa": 7, "mmlu": 7, "ifeval": 32}
pool = [json.loads(line) for line in open(sys.argv[1])]
used = [json.loads(line) for line in open(sys.argv[2])]
used_ids = {j["id"] for j in used}
per_family = collections.Counter(j["family"] for j in used)
rest = collections.defaultdict(list)
for item in pool:
    if item["id"] not in used_ids:
        rest[item["family"]].append(item)
out = []
for family, n in SUBSET.items():
    fresh = rest[family] if family == "ifeval" else rest[family][:per_family[family]]
    out += [fresh[int(i * len(fresh) / n)] for i in range(n)]
with open(sys.argv[3], "w") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(len(out), "prompts:", dict(collections.Counter(j["family"] for j in out)))
