"""Format-and-finishing census (measurement only, 2026-11-16): build the prompt list on CPU.

Three slices, written to ../census/items.jsonl in the format mine.py --items reads:
  exact  100 questions x 3 answer-only wordings; gold is a bare integer (GSM8K train) or an option letter (ARC, MMLU aux,
         CommonsenseQA train). None of the questions is among the 3,000 mined training prompts.
  coding 150 MBPP test tasks wrapped by the platform's own coding_prompt (JSON files object, solution.py).
  trunc  150 held-out boxed-math prompts from probe sets 1 and 2 on which the leader or fc step 60 hit the token cap most.
The fourth slice (single-turn tool calling) runs through the official harness, see census.sh.
"""
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import grpo
from delta_cli.public_eval.custom_tasks import coding_prompt

OUT = Path("../census"); OUT.mkdir(exist_ok=True)
key_of = lambda prompt: hashlib.sha1(prompt.encode()).hexdigest()[:16]
rng = random.Random(20261116)

WORDINGS = {
    "w1": lambda q, kind: q + "\n\nGive only the final answer, with no explanation or extra text.",
    "w2": lambda q, kind: q + "\n\nRespond with just " + ("the number" if kind == "num" else "the letter of the correct option") + " and nothing else.",
    "w3": lambda q, kind: "Answer the question below. Your entire response must be " + ("a single number" if kind == "num" else "a single letter")
                          + " and nothing more.\n\n" + q,
}

if __name__ == "__main__":
    mined = {json.loads(l)["key"] for l in open("mined/leader.jsonl")}
    items = []

    # ---- exact slice
    gsm = [x for x in grpo.math_pool() if x["src"] == "gsm8k" and re.fullmatch(r"-?\d+", x["answer"]) and key_of(x["prompt"]) not in mined]
    cho = [x for x in grpo.choice_pool(broad=True) if x["src"] in ("arc", "mmlu_aux", "csqa") and key_of(x["prompt"]) not in mined]
    base = [("num", x) for x in rng.sample(gsm, 50)]
    for src, n in (("arc", 17), ("mmlu_aux", 17), ("csqa", 16)):
        base += [("choice", x) for x in rng.sample([c for c in cho if c["src"] == src], n)]
    for i, (kind, x) in enumerate(base):
        q = x["prompt"][: -len(grpo.SUFFIX)] if kind == "num" else x["prompt"][: -len(grpo.CHOICE_SUFFIX)]
        assert q and q != x["prompt"], "suffix not found"
        for w, fn in WORDINGS.items():
            p = fn(q, kind)
            items.append({"prompt": p, "scorer": "exact", "answer": x["answer"], "src": x["src"], "pool": "exact", "key": key_of(p),
                          "slice": "exact", "kind": kind, "wording": w, "base": i})

    # ---- coding slice (same first 150 MBPP test tasks the broad evaluator scored in the plain format)
    import gzip
    HEAD = "Write Python code solving this task. Return code in a Python code block.\n"; MID = "\nThe code must pass:\n"
    docs = {}
    with gzip.open("../broad/leader/mbpp_0_500/mbpp.raw.jsonl.gz", "rt") as g:   # the harness's own requests: task text and tests
        for line in g:
            c = json.loads(line)["request"]["messages"][0]["content"]
            assert c.startswith(HEAD) and MID in c
            text, tests = c[len(HEAD):].split(MID, 1)
            docs[key_of(c)] = {"text": text, "test_list": [t for t in tests.split("\n") if t.strip()]}
    n_code = 0
    for k in sorted(docs):
        if n_code == 150:
            break
        d = docs[k]
        m = re.match(r"\s*assert\s+\(?\s*(?:not\s+)?(?:set\(|sorted\(|tuple\(|list\(|abs\(|round\(|math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(", d["test_list"][0])
        if not m:
            continue
        task = "Write Python code solving this task.\n" + d["text"] + MID + "\n".join(d["test_list"])
        p = coding_prompt(task, {"solution.py": ""})
        items.append({"prompt": p, "scorer": "exact", "answer": "", "src": "mbpp", "pool": "coding", "key": key_of(p), "slice": "coding",
                      "plain_key": k, "func": m.group(1), "test_list": d["test_list"], "test_setup_code": ""})
        n_code += 1

    # ---- truncation-prone boxed math (held out from training; chosen by cap hits of leader + fc60, 8 samples)
    def load(p):
        out = {}
        for l in open(p):
            r = json.loads(l); r["item"] = r["item"] if isinstance(r["item"], dict) else ast.literal_eval(r["item"]); out[r["key"]] = r
        return out
    cand = []
    for a, b in (("probe/leader.jsonl", "probe/fc_s60.jsonl"), ("probe/leader_p2.jsonl", "probe/fc_s60_p2.jsonl")):
        A, B = load(a), load(b)
        for k, r in A.items():
            if r["pool"] != "math" or k not in B:
                continue
            hits = sum(t >= 8192 for t in r["tokens"]) + sum(t >= 8192 for t in B[k]["tokens"])
            if hits:
                cand.append((hits, k, r["item"]))
    cand.sort(key=lambda x: (-x[0], x[1]))
    for hits, k, it in cand[:150]:
        items.append({**it, "pool": "trunc", "slice": "trunc", "cap_hits_of_8": hits})

    assert len({i["key"] for i in items}) == len(items), "duplicate keys"
    with open(OUT / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({"key": it["key"], "item": it}) + "\n")
    from collections import Counter
    print(len(items), "prompts:", dict(Counter(i["slice"] for i in items)), "| truncation-prone candidates available:", len(cand),
          "| cap hits of the chosen 150: min", min(i["cap_hits_of_8"] for i in items if i["slice"] == "trunc"))
    for s in ("exact", "coding", "trunc"):
        ex = next(i for i in items if i["slice"] == s)
        print(f"--- {s} example:\n{ex['prompt'][:600]}\n    gold: {ex.get('answer')!r}")
