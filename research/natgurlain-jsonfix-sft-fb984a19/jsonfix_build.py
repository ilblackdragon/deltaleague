"""JSON-files contract fix, step 1 (CPU): prompts on which fc step 60 will be sampled to make its own training data.

Tasks: MBPP train + validation + prompt splits only (474 tasks; the 500 test tasks are never used for training).
  ../jsonfix/gen_items.jsonl     JSON-contract prompts (platform's coding_prompt): about 410 training tasks + 60 dev tasks
  ../jsonfix/dev_items.jsonl     the 60 dev tasks alone (to pick a checkpoint without touching the census slice)
  ../jsonfix/replay_items.jsonl  prompts in OTHER formats whose current answers the fine-tune must keep: the same training
                                 tasks in the plain code-block format, pool prompts (boxed math, choice, ifeval, logic),
                                 and answer-only prompts (questions that are not in the census)
"""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import grpo
from datasets import load_dataset
from delta_cli.public_eval.custom_tasks import coding_prompt
from delta_cli.public_eval.coding import prompt as plain_prompt
from census_build import WORDINGS  # noqa: E402  (importing rebuilds nothing: guarded below)

OUT = Path("../jsonfix"); OUT.mkdir(exist_ok=True)
key_of = lambda p: hashlib.sha1(p.encode()).hexdigest()[:16]
rng = random.Random(20261117)
FUNC = re.compile(r"\s*assert\s+\(?\s*(?:not\s+)?(?:set\(|sorted\(|tuple\(|list\(|abs\(|round\(|math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(")

ds = load_dataset("google-research-datasets/mbpp", "full")
tasks = []
for split in ("train", "validation", "prompt"):
    for d in ds[split]:
        m = FUNC.match(d["test_list"][0])
        if m:
            tasks.append({"task_id": d["task_id"], "text": d["text"], "test_list": list(d["test_list"]), "test_setup_code": d.get("test_setup_code") or "", "func": m.group(1)})
tasks.sort(key=lambda t: int(t["task_id"])); rng.shuffle(tasks)
dev, train = tasks[:60], tasks[60:]

def json_item(t, split, starter):
    task = "Write Python code solving this task.\n" + t["text"] + "\nThe code must pass:\n" + "\n".join(t["test_list"])
    p = coding_prompt(task, starter)
    return {"prompt": p, "scorer": "exact", "answer": "", "src": "mbpp_" + split, "pool": "json", "key": key_of(p), "slice": "json", "split": split,
            "starter": starter, "fname": next(iter(starter)), **{k: t[k] for k in ("task_id", "func", "test_list", "test_setup_code")}}

gen = []
for i, t in enumerate(train):
    r = i % 20
    if r < 14:
        starter = {"solution.py": ""}
    elif r < 17:
        starter = {"solution.py": f"def {t['func']}(*args):\n    raise NotImplementedError\n"}
    else:
        starter = {("main.py", "answer.py", "task.py")[r - 17]: ""}
    gen.append(json_item(t, "train", starter))
devs = [json_item(t, "dev", {"solution.py": ""}) for t in dev]

replay = []
for t in train[:150]:
    p = plain_prompt("mbpp", t)
    replay.append({"prompt": p, "scorer": "exact", "answer": "", "src": "mbpp_plain", "pool": "replay_code", "key": key_of(p), "slice": "replay", **{k: t[k] for k in ("task_id", "func", "test_list", "test_setup_code")}})
pool = [json.loads(l) for l in open("mined/mix.jsonl")]
for it in rng.sample(pool, 200):
    replay.append({**it, "pool": "replay_" + it["pool"], "slice": "replay", "key": it.get("key") or key_of(it["prompt"])})
census = {json.loads(l)["key"] for l in open("../census/items.jsonl")}
mined = {json.loads(l)["key"] for l in open("mined/leader.jsonl")}
gsm = [x for x in grpo.math_pool() if x["src"] == "gsm8k" and re.fullmatch(r"-?\d+", x["answer"])]
cho = [x for x in grpo.choice_pool(broad=True) if x["src"] in ("arc", "mmlu_aux", "csqa")]
n = 0
for kind, src, suffix in [("num", gsm, grpo.SUFFIX)] * 50 + [("choice", cho, grpo.CHOICE_SUFFIX)] * 50:
    x = rng.choice(src); q = x["prompt"][: -len(suffix)]; w = rng.choice(list(WORDINGS)); p = WORDINGS[w](q, kind)
    if any(key_of(WORDINGS[v](q, kind)) in census for v in WORDINGS):
        continue
    replay.append({"prompt": p, "scorer": "exact", "answer": x["answer"], "src": x["src"], "pool": "replay_exact", "key": key_of(p), "slice": "replay"})
seen = set(); replay = [r for r in replay if not (r["key"] in seen or seen.add(r["key"]))]

def write(name, items):
    with open(OUT / name, "w") as f:
        for it in items:
            f.write(json.dumps({"key": it["key"], "item": it}) + "\n")
write("gen_items.jsonl", gen + devs); write("dev_items.jsonl", devs); write("replay_items.jsonl", replay)
print("tasks", len(tasks), "| JSON prompts: train", len(gen), "dev", len(devs), "| starter variants", dict(Counter(("stub" if g["starter"][g["fname"]] else "empty") + ":" + g["fname"] for g in gen)))
print("replay prompts", len(replay), dict(Counter(r["pool"] for r in replay)))
print("--- variant example:\n" + gen[14]["prompt"][-420:])
