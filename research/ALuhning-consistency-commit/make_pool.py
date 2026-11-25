"""Filter practice problems to those the parent solves at least once (correct, thinking closed, stopped),
re-scored with the trainer's checker. Usage: make_pool.py PROBE.jsonl ITEMS.jsonl OUT_POOL.jsonl"""
import json, sys
from grpo_gates import is_correct
rows = [json.loads(l) for l in open(sys.argv[1])]
items = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
ok = {}
for r in rows:
    t = r.get("text", ""); final = t.split("</think>", 1)[1] if "</think>" in t else ""
    good = r["finish_reason"] == "stop" and is_correct(r["bench"], final, items[r["id"]]["answer"])
    ok[r["id"]] = ok.get(r["id"], 0) + good
pool = [items[i] for i, n in ok.items() if n > 0]
with open(sys.argv[3], "w") as f:
    for it in pool: f.write(json.dumps(it) + "\n")
print(f"generations {len(rows)}, problems {len(ok)}, solved at least once {len(pool)}, success rate {sum(ok.values())/len(rows):.3f}")
