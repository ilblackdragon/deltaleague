"""Step 2: pick SFT traces from the generated samples.

Keeps a trace only if:
  - it is correct, finished, and closes </think>;
  - the answer after </think> is short (<= 300 tokens for MC, <= 600 for math) and contains \\boxed{};
  - the total length is <= MAX_TOTAL_TOKENS, which leaves headroom under the 8192 cap;
  - its tail does not loop.
Per item it picks ONE trace:
  - multiple choice: prefer 3-in-1 (it deliberates longer), else the leader;
  - math: the leader's traces;
  - among the preferred candidates, the median-length one (not the shortest).

Usage: python select_traces.py GEN_DIR OUT.jsonl [--leader cd0b95bb] [--deliberate a9340e14]
"""
import argparse
import collections
import gzip
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("gen_dir")
ap.add_argument("out")
ap.add_argument("--items", default=None, help="items json (default GEN_DIR/train_items.json)")
ap.add_argument("--leader", default="cd0b95bb")
ap.add_argument("--deliberate", default="a9340e14")
ap.add_argument("--tokenizer", default=str(Path(__file__).parent / "model-hf"))
ap.add_argument("--max-answer-tokens-mc", type=int, default=300)
ap.add_argument("--max-answer-tokens-math", type=int, default=600)
ap.add_argument("--max-total-tokens", type=int, default=6000)
ap.add_argument("--min-think-tokens-mc", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

gen = Path(args.gen_dir)
items = {it["id"]: it for it in json.load(open(args.items or gen / "train_items.json"))}
tok = AutoTokenizer.from_pretrained(args.tokenizer)
SUFFIX = {"math": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
          "mc": "\n\nAnswer with the letter of the correct option inside \\boxed{}."}


def loops(text, n=40):
    t = text[-3000:]
    c = collections.Counter(t[i:i + n] for i in range(0, max(0, len(t) - n), 10))
    return bool(c) and max(c.values()) >= 4


def load(v):
    p = gen / f"{v}.jsonl.gz"
    return [json.loads(l) for l in gzip.open(p, "rt")] if p.exists() else []


stats = collections.Counter()
cands = collections.defaultdict(list)  # id -> [(pref, rec)]
for v, pref_mc, pref_math in [(args.deliberate, 0, 9), (args.leader, 1, 0)]:
    for r in load(v):
        stats[f"{v}_total"] += 1
        it = items.get(r["id"])
        if it is None or not r["correct"] or not r["finished"] or "</think>" not in r["text"]:
            continue
        think, _, ans = r["text"].partition("</think>")
        if "\\boxed{" not in ans or loops(r["text"]):
            stats["reject_box_or_loop"] += 1; continue
        a_len = len(tok.encode(ans))
        t_len = len(tok.encode(think))
        max_a = args.max_answer_tokens_mc if it["kind"] == "mc" else args.max_answer_tokens_math
        if a_len > max_a or r["len"] > args.max_total_tokens:
            stats["reject_length"] += 1; continue
        if it["kind"] == "mc" and t_len < args.min_think_tokens_mc:
            stats["reject_short_think"] += 1; continue
        pref = pref_mc if it["kind"] == "mc" else pref_math
        cands[r["id"]].append((pref, r, t_len, a_len, v))
        stats[f"{v}_kept"] += 1

rng = random.Random(args.seed)
out = []
per_src = collections.defaultdict(list)
for iid, cs in cands.items():
    best = min(c[0] for c in cs)
    cs = sorted([c for c in cs if c[0] == best], key=lambda c: c[1]["len"])
    pref, r, t_len, a_len, v = cs[len(cs) // 2]
    it = items[iid]
    prompt = tok.apply_chat_template([{"role": "user", "content": it["q"] + SUFFIX[it["kind"]]}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
    rec = {"id": iid, "src": it["src"], "from": v,
           "prompt": prompt, "completion": r["text"] + "<|im_end|>",
           "len": r["len"], "think_tokens": t_len, "answer_tokens": a_len}
    out.append(rec)
    per_src[it["src"]].append(rec)

rng.shuffle(out)
with open(args.out, "w") as f:
    for rec in out:
        f.write(json.dumps(rec) + "\n")

print(dict(stats))
print(f"selected {len(out)} traces -> {args.out}")
for s, rs in sorted(per_src.items()):
    lens = sorted(r["len"] for r in rs)
    frm = collections.Counter(r["from"] for r in rs)
    print(f"  {s:15} n={len(rs):4}  median_len={lens[len(lens) // 2]:5}  "
          f"median_answer={sorted(r['answer_tokens'] for r in rs)[len(rs) // 2]:4}  from={dict(frm)}")
