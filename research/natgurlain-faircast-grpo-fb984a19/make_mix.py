"""Training pool for the Fair Cast runs: every tail prompt (select_tail.py) plus a third as many of the other
mined prompts, so a quarter of the pool is prompts the parent already answers consistently and briefly."""
import json
import random
import sys
from collections import Counter

out = sys.argv[1] if len(sys.argv) > 1 else "mined/mix.jsonl"
rows = [json.loads(l) for l in open("mined/leader.jsonl") if l.strip()]
tail = [json.loads(l) for l in open("mined/tail.jsonl") if l.strip()]
tk = {json.dumps(t, sort_keys=True) for t in tail}
rest = [r["item"] for r in rows if json.dumps(r["item"], sort_keys=True) not in tk]
random.Random(7).shuffle(rest)
extra = rest[: len(tail) // 3]
with open(out, "w") as f:
    for it in tail + extra:
        f.write(json.dumps(it) + "\n")
print(len(tail), len(rest), len(extra), Counter(i["pool"] for i in tail + extra))
