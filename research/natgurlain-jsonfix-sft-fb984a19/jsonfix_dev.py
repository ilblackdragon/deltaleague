"""Dev-set check for the JSON-files fix (CPU): valid-response and pass rate on the 60 held-out training-split tasks.
usage (.venv-public python, DELTA_WASI_ROOT=../.wasi): jsonfix_dev.py FILE.jsonl [FILE.jsonl ...]   (mine.py outputs with text)"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import census_score as cs
from delta_cli.public_eval.generation import final_answer

for path in sys.argv[1:]:
    jobs = []
    for line in open(path):
        r = json.loads(line); it = r["item"]
        if it.get("slice") != "json" or it.get("split") != "dev":
            continue
        jobs += [(it, final_answer(t), tok) for t, tok in zip(r["text"], r["tokens"])]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: cs.code_result(j[0], j[1]), jobs))
    n = len(res)
    print(f"{path}: {n} samples  valid {100 * sum(r[0] for r in res) / n:5.1f}%  strict JSON {100 * sum(r[1] for r in res) / n:5.1f}%  "
          f"passes tests {100 * sum(r[2] for r in res) / n:5.1f}%  thinking closed {100 * sum(bool(j[1]) for j in jobs) / n:5.1f}%  capped {100 * sum(j[2] >= 8192 for j in jobs) / n:4.1f}%")
