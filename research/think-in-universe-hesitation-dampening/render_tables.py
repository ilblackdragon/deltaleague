#!/usr/bin/env python3
"""Render markdown evidence tables from local_eval/mc_eval reports."""
from __future__ import annotations

import json
import sys


def mc_table(base, cand):
    b = json.loads(open(base).read())
    c = json.loads(open(cand).read())
    lines = ["| benchmark | baseline acc_norm | candidate acc_norm | delta |",
             "| --- | ---: | ---: | ---: |"]
    names = sorted(set(b["components"]) & set(c["components"]))
    for name in names:
        if b["components"][name] is None:
            continue
        d = c["components"][name] - b["components"][name]
        lines.append(f"| {name} | {b['components'][name]:.2f} | {c['components'][name]:.2f} | {d:+.2f} |")
    lines.append(f"| **equal-weight mean ({len(names)})** | **{b['mean']:.2f}** | **{c['mean']:.2f}** | **{c['mean'] - b['mean']:+.2f}** |")
    return "\n".join(lines)


def gen_table(base, cand, compare):
    b = json.loads(open(base).read())
    c = json.loads(open(cand).read())
    cmp_ = json.loads(open(compare).read())
    lines = ["| benchmark | baseline | candidate | delta |",
             "| --- | ---: | ---: | ---: |"]
    for name in sorted(b["benchmarks"]):
        bb, cc = b["benchmarks"][name], c["benchmarks"][name]
        lines.append(f"| {name} (n={bb['n']}) | {bb['score']:.1f} | {cc['score']:.1f} | {cc['score'] - bb['score']:+.1f} |")
    lines += ["",
              f"Paired overall difference: **{cmp_['difference'] * 100:+.1f} points** "
              f"(95% bootstrap CI {cmp_['difference_ci95'][0] * 100:+.1f} to {cmp_['difference_ci95'][1] * 100:+.1f}), "
              f"wins/losses/ties {cmp_['wins']}/{cmp_['losses']}/{cmp_['ties']}, sign-test p={cmp_['sign_test_p']:.3f}.",
              "",
              "| metric | baseline | candidate |",
              "| --- | ---: | ---: |",
              f"| truncation rate | {cmp_['truncation']['baseline']:.3f} | {cmp_['truncation']['candidate']:.3f} |",
              f"| mean output tokens | {cmp_['mean_tokens']['baseline']:.0f} | {cmp_['mean_tokens']['candidate']:.0f} |",
              f"| hesitation words in final answer | {cmp_['hesitation_mentions_in_final_answer']['baseline']:.2f} | {cmp_['hesitation_mentions_in_final_answer']['candidate']:.2f} |"]
    return "\n".join(lines)


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "mc":
        print(mc_table(*sys.argv[2:4]))
    else:
        print(gen_table(*sys.argv[2:5]))
