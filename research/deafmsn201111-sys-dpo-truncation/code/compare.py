"""Paired comparison of proxy eval records (same 500 items) against stored baselines.

Usage: python compare.py CANDIDATE.jsonl.gz [more...]
Baselines: leader = runs/rerun1/calib/cd0b95bb.jsonl.gz (rescored), cand_d10 = runs/train1/train/eval_cand_d10.jsonl.gz
"""
import gzip, json, random, sys
from rescore_lib import letter

items = {i["id"]: i for i in json.load(open("runs/rerun1/calib/items.json"))}


def load(p):
    recs = {r["id"]: r for r in map(json.loads, gzip.open(p, "rt"))}
    for i, r in recs.items():  # uniform scoring: format-tolerant MC, stored math scores
        if items[i]["kind"] == "mc":
            r["correct"] = int(r["finished"] and "</think>" in r["text"] and letter(r["text"].split("</think>")[-1]) == items[i]["gold"])
        r["trunc"] = int(not r["finished"]); r["trunc_after"] = int(not r["finished"] and "</think>" in r["text"])
    return recs


def ci(d):
    rng = random.Random(0); n = len(d)
    b = sorted(sum(rng.choice(d) for _ in range(n)) / n * 100 for _ in range(4000))
    return f"{100 * sum(d) / n:+5.1f} [{b[100]:+.1f}, {b[3899]:+.1f}]"


base = {"leader": load("runs/rerun1/calib/cd0b95bb.jsonl.gz"), "cand_d10": load("runs/train1/train/eval_cand_d10.jsonl.gz")}
srcs = ["gsm8k", "math500", "mmlu_pro", "arc_challenge", "commonsense_qa"]
for p in sys.argv[1:]:
    C = load(p); ids = sorted(C)
    print(f"\n== {p}  acc {100 * sum(C[i]['correct'] for i in ids) / len(ids):.1f}  trunc {100 * sum(C[i]['trunc'] for i in ids) / len(ids):.1f}"
          f"  trunc_after_think {100 * sum(C[i]['trunc_after'] for i in ids) / len(ids):.1f}  tokens {sum(C[i]['len'] for i in ids)}")
    print("   by src:", {s: round(100 * sum(C[i]['correct'] for i in ids if C[i]['src'] == s) / 100, 1) for s in srcs})
    for name, B in base.items():
        print(f"   vs {name:8} acc {ci([C[i]['correct'] - B[i]['correct'] for i in ids])}  trunc {ci([C[i]['trunc'] - B[i]['trunc'] for i in ids])}"
              f"  math500 acc {ci([C[i]['correct'] - B[i]['correct'] for i in ids if C[i]['src'] == 'math500'])}")
