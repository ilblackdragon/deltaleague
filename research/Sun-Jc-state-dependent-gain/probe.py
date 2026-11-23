"""Linear probes on stored activations: can the model's state tell how long it has been generating?

Fits ridge probes on a train split of completions and evaluates on held-out completions
(the split is by completion, never by position, so a probe cannot memorise a trace).

Per site:
  length   regress the generated step (thousands of tokens) from all states
  late     the same score as a detector of "step >= HI" against "step <= LO": AUC, and the
           operating points that matter for a deadline gate: for a threshold that catches
           50/80/95% of late states, the share of early states and of whole normal
           completions (any state) that would also cross it
  phase    thinking vs answering (states of completions that finished normally)

usage: python probe.py ACTS_DIR [ACTS_DIR ...] [--lam 10] [--lo 4000] [--hi 6000] [--sites a,b] [--save probes.pt]
"""
import glob
import os
import sys

import torch

args = sys.argv[1:]
arg = lambda name, default: type(default)(args[args.index(name) + 1]) if name in args else default
lam, lo, hi = arg("--lam", 10.0), arg("--lo", 4000), arg("--hi", 6000)
save, only_sites = arg("--save", ""), arg("--sites", "")
dirs, skip = [], False
for a in args:
    if skip:
        skip = False
    elif a.startswith("--"):
        skip = True
    else:
        dirs.append(a)

rows, metas, X = [], [], {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "acts-*.pt"))):
        part = torch.load(f)
        m = part["meta"].clone()
        m[:, 0] += len(rows)
        metas.append(m)
        rows.extend(part["rows"])
        for site, x in part["X"].items():
            if not only_sites or site in only_sites.split(","):
                X.setdefault(site, []).append(x)
meta = torch.cat(metas)
row_idx, step, since = meta[:, 0].long(), meta[:, 1].double(), meta[:, 2]
n_rows = len(rows)
g = torch.Generator().manual_seed(0)
capped_row = torch.tensor([r["outcome"] in ("runaway", "overflow") for r in rows])
# Stratified split by completion: a third of the capped and a third of the normal completions are held out.
test_rows = torch.zeros(n_rows, dtype=torch.bool)
for group in (capped_row, ~capped_row):
    idx = group.nonzero().flatten()
    idx = idx[torch.randperm(len(idx), generator=g)]
    test_rows[idx[: len(idx) // 3]] = True
is_test = test_rows[row_idx]
thinking = since < 0
is_capped = capped_row[row_idx]
fam = [r["family"] for r in rows]
print(f"{n_rows} completions ({int(capped_row.sum())} capped), {meta.shape[0]:,} positions; "
      f"held out: {int((test_rows & capped_row).sum())} capped + {int((test_rows & ~capped_row).sum())} normal completions")
print(f"states with step >= {hi}: train {int((~is_test & (step >= hi)).sum())}, test {int((is_test & (step >= hi)).sum())}; "
      f"states with step <= {lo}: train {int((~is_test & (step <= lo)).sum())}, test {int((is_test & (step <= lo)).sum())}")


def ridge(x, y, lam):
    """Ridge with an intercept; inputs standardised with train statistics."""
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mu) / sd
    w = torch.linalg.solve(z.T @ z + lam * z.shape[0] / 1000 * torch.eye(z.shape[1], dtype=z.dtype), z.T @ (y - y.mean()))
    return w / sd, float(y.mean() - (w / sd) @ mu)


def auc(scores, labels):
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    pos = labels.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def operating_points(score, tag):
    """How selective is `score` as a late detector on held-out completions?"""
    late, early = is_test & (step >= hi), is_test & (step <= lo)
    if int(late.sum()) == 0 or int(early.sum()) == 0:
        return
    print(f"  {tag}: AUC late-vs-early {auc(score[late | early], (step[late | early] >= hi).double()):.4f}")
    normal_rows = (test_rows & ~capped_row).nonzero().flatten().tolist()
    row_max = {r: float(score[(row_idx == r)].max()) for r in normal_rows if int((row_idx == r).sum())}
    for catch in (0.5, 0.8, 0.95):
        thr = float(torch.quantile(score[late], 1 - catch))
        early_fp = float((score[early] > thr).double().mean())
        rows_fp = sum(v > thr for v in row_max.values()) / max(len(row_max), 1)
        print(f"     threshold catching {int(catch * 100):2d}% of late states: {100 * early_fp:7.3f}% of early states cross it, "
              f"{100 * rows_fp:5.1f}% of normal completions cross it somewhere")


saved = {}
for site, parts in X.items():
    x = torch.cat(parts).double()
    print(f"\n=== {site} ===")
    tr, te = ~is_test, is_test
    w, b = ridge(x[tr], step[tr] / 1000, lam)
    pred = x @ w + b
    resid = pred[te] - step[te] / 1000
    r2 = 1 - float(resid.pow(2).sum() / (step[te] / 1000 - (step[te] / 1000).mean()).pow(2).sum())
    print(f"length probe: held-out R^2 {r2:.3f}, RMSE {float(resid.pow(2).mean().sqrt()) * 1000:.0f} tokens")
    for a, bnd in [(0, 500), (500, 1500), (1500, 3000), (3000, 4500), (4500, 6000), (6000, 7000), (7000, 8200)]:
        sel = te & (step >= a) & (step < bnd)
        if int(sel.sum()) > 0:
            q = torch.quantile(pred[sel], torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=pred.dtype)) * 1000
            print(f"  true step {a:5d}-{bnd:5d}: n={int(sel.sum()):6d}  predicted p1/p5/p50/p95/p99 = "
                  f"{q[0]:6.0f} / {q[1]:6.0f} / {q[2]:6.0f} / {q[3]:6.0f} / {q[4]:6.0f}")
    operating_points(pred, "length score as late detector")
    # A direct late-vs-early discriminant (ridge on +-1 labels), trained only on those two groups.
    grp = tr & ((step >= hi) | (step <= lo))
    wl, bl = ridge(x[grp], (step[grp] >= hi).double() * 2 - 1, lam)
    late_score = x @ wl + bl
    operating_points(late_score, "late-vs-early discriminant ")
    for family in sorted(set(fam)):
        sel = te & torch.tensor([fam[i] == family for i in row_idx.tolist()])
        if int(sel.sum()) > 50:
            rr = pred[sel] - step[sel] / 1000
            print(f"  family {family:7s}: n={int(sel.sum()):6d}  length bias {float(rr.mean()) * 1000:+6.0f}  RMSE {float(rr.pow(2).mean().sqrt()) * 1000:5.0f} tokens")
    normal = ~is_capped
    if int((normal & ~thinking).sum()) > 0 and int((normal & thinking).sum()) > 0:
        wp, bp = ridge(x[tr & normal], thinking[tr & normal].double() * 2 - 1, lam)
        ph = x @ wp + bp
        acc = float(((ph[te & normal] > 0) == thinking[te & normal]).double().mean())
        print(f"phase probe (normal completions): held-out accuracy {acc:.4f}; mean score thinking {float(ph[te & normal & thinking].mean()):+.2f}, "
              f"answering {float(ph[te & normal & ~thinking].mean()):+.2f}")
        saved[site] = {"phase_w": wp.float(), "phase_b": bp}
    saved.setdefault(site, {}).update({"length_w": w.float(), "length_b": b, "late_w": wl.float(), "late_b": bl})
    del x

if save:
    torch.save(saved, save)
    print("saved probes to", save)
