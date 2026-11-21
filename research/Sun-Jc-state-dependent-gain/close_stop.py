"""Does an edit change when the model closes its thinking or ends its turn?

Reads the output of `edit_eval.py collect` and compares, under the evaluation sampler, the
probability of </think> and of the end-of-turn token in the parent and in the edited model:
  - </think> at thinking states where the parent gives it more than --min-p (the states where
    closing is a live option),
  - the end-of-turn token at the last state of every finished completion.
Completions of the families in --changed (those the edit is meant to change) are reported
separately from the rest, which the edit must leave alone.

usage: python close_stop.py OUT [OUT ...] [--changed ifeval] [--min-p 0.02]
"""
import argparse

import torch

from edit_eval import load_parts

THINK_CLOSE, END_OF_TURN = 248069, 248046
ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--changed", default="ifeval")
ap.add_argument("--min-p", type=float, default=0.02)
args = ap.parse_args()

d, rows, watch = load_parts(args.dirs)
row, since, target = d["meta"][:, 0].long(), d["meta"][:, 2].long(), d["meta"][:, 3].long()
changed = set(args.changed.split(","))
is_changed = torch.tensor([r["family"] in changed for r in rows])[row]
thinking = since < 0
p_close_b, p_close_e = (d[k][:, watch.index(THINK_CLOSE)].double() for k in ("p_watch_base", "p_watch_edit"))
p_end_b, p_end_e = (d[k][:, watch.index(END_OF_TURN)].double() for k in ("p_watch_base", "p_watch_edit"))
families = {}
for r in rows:
    families[r["family"]] = families.get(r["family"], 0) + 1
print(f"{len(rows)} completions {dict(sorted(families.items()))}, {len(row):,} positions")


def ratio(name, sel, pb, pe):
    if int(sel.sum()) == 0:
        print(f"   {name:58s} no such state")
        return
    q = pe[sel] / pb[sel]
    print(f"   {name:58s} n={int(sel.sum()):5d}  edited/parent {q.mean():.3f} +- {q.std():.3f}  (min {q.min():.3f}, max {q.max():.3f})")


for label, group in (("families the edit must leave alone", ~is_changed), (f"families the edit changes ({', '.join(sorted(changed))})", is_changed)):
    print(label + ":")
    ratio(f"P(</think>) at thinking states where the parent's > {args.min_p}", group & thinking & (p_close_b > args.min_p), p_close_b, p_close_e)
    ratio("P(end of turn) at the last state of a finished completion", group & ~thinking & (target == END_OF_TURN), p_end_b, p_end_e)
    for name, sel in (("thinking", group & thinking), ("answer", group & ~thinking)):
        if int(sel.sum()):
            print(f"   total variation per token, {name:9s} n={int(sel.sum()):6d}  mean {d['tv'][sel].double().mean():.4f}")
