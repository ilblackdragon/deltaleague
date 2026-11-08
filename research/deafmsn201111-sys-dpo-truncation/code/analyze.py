"""Failure analysis of proxy generations: where does a model lose points?

Usage: python analyze.py runs/rerun1/calib cd0b95bb [a9340e14]
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
items = {it["id"]: it for it in json.load(open(d / "items.json"))}


def load(v):
    return {r["id"]: r for r in map(json.loads, gzip.open(d / f"{v}.jsonl.gz", "rt"))}


def boxed(text):
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + 7:k - 1])
        i = k


def tail_loop(text, n=40):
    """Fraction of the last 2000 chars covered by a repeated n-char chunk."""
    t = text[-2000:]
    seen = collections.Counter(t[i:i + n] for i in range(0, len(t) - n, 10))
    return max(seen.values()) if seen else 0


def classify(r):
    it = items[r["id"]]
    t = r["text"]
    if not r["finished"]:
        if "</think>" in t:
            return "trunc_in_answer"
        return "trunc_loop" if tail_loop(t) >= 5 else "trunc_thinking"
    if "</think>" not in t:
        return "no_think_close"
    if r["correct"]:
        return "correct"
    ans = t.split("</think>")[-1]
    b = boxed(ans)
    if not b:
        return "wrong_no_boxed"
    if it["kind"] == "mc" and not re.fullmatch(r"\s*(\\text\{)?\(?[A-J]\)?\}?\.?\s*", b[-1]):
        return "wrong_mc_format"
    return "wrong_answer"


for v in sys.argv[2:]:
    recs = load(v)
    print(f"\n===== {v}  n={len(recs)}")
    by = collections.defaultdict(collections.Counter)
    lens = collections.defaultdict(list)
    for r in recs.values():
        c = classify(r)
        by[r["src"]][c] += 1
        by["ALL"][c] += 1
        lens[(r["src"], r["correct"])].append(r["len"])
    cats = ["correct", "wrong_answer", "wrong_mc_format", "wrong_no_boxed", "no_think_close",
            "trunc_thinking", "trunc_loop", "trunc_in_answer"]
    print(f"{'src':15}" + "".join(f"{c[:15]:>16}" for c in cats))
    for s, cnt in by.items():
        print(f"{s:15}" + "".join(f"{cnt[c]:>16}" for c in cats))
    print("median tokens (correct / wrong):")
    for s in sorted({k[0] for k in lens}):
        def med(x):
            return sorted(x)[len(x) // 2] if x else None
        print(f"  {s:15} {med(lens[(s, 1)])} / {med(lens[(s, 0)])}")

    # Examples of MC format problems and finished-wrong commonsense answers
    ex = [r for r in recs.values() if classify(r) == "wrong_mc_format"][:5]
    for r in ex:
        print("  MC-FORMAT", r["src"], "gold", items[r["id"]]["gold"], "->",
              boxed(r["text"].split("</think>")[-1])[-1][:80])

if len(sys.argv) > 3:
    a, b = load(sys.argv[2]), load(sys.argv[3])
    print(f"\n===== paired {sys.argv[2]} vs {sys.argv[3]}")
    for s in sorted({r["src"] for r in a.values()}):
        ids = [i for i in a if a[i]["src"] == s]
        only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
        only_b = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
        print(f"  {s:15} only {sys.argv[2]}: {only_a:3}   only {sys.argv[3]}: {only_b:3}")
