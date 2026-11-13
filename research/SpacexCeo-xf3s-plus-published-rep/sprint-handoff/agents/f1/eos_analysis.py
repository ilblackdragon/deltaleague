"""EOS lever: along answer phases (TF'd through base), p(EOS) under the stack with EOS row scale f; counts expected
early terminations in normal answers (before the actual EOS) vs. expected terminations inside overflow answers."""
import sys, os, numpy as np
d = sys.argv[1]; EOS = 248044; PEN = 1.5
def p_eos(z, topv, topi, tseen, eseen, g, f):
    # candidate set: top8 others + EOS (two-level approx of the top-k/top-p sampler: EOS is sampled iff it is in the nucleus)
    Lo = g * topv - PEN * tseen; Lo[topi == EOS] = -np.inf
    Le = g * f * z - PEN * eseen
    L = np.concatenate([Lo, Le[:, None]], 1); L = L - L.max(1, keepdims=True); p = np.exp(L); p /= p.sum(1, keepdims=True)
    pe = p[:, -1]
    # top-p 0.95: EOS is excluded if the others' cumulative mass above it already reaches 0.95
    others_sorted = -np.sort(-p[:, :-1], 1); above = (others_sorted * (others_sorted > pe[:, None])).sum(1)
    pe = np.where(above >= 0.95, 0.0, pe)
    return pe
for f in [1.0, 1.1, 1.2, 1.3, 1.5]:
    early_norm, n_norm, hit_over, n_over, len_norm = 0.0, 0, 0.0, 0, 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.npz'): continue
        a = np.load(fn if os.path.isabs(fn) else os.path.join(d, fn)); z = a['ze'].astype(np.float64)
        pe = p_eos(z, a['topv'].astype(np.float64), a['topi'], a['tseen'], a['eseen'], 1.4, f)
        if a['trunc']:
            n_over += 1; hit_over += 1 - np.prod(1 - pe)              # P(answer ends somewhere before the limit)
        else:
            n_norm += 1; early_norm += 1 - np.prod(1 - pe[:-1]); len_norm += len(z)   # P(ends before the real EOS)
    print(f'EOS x{f}: normal answers n={n_norm}: P(cut early) mean {early_norm/max(1,n_norm):.3f} | overflow answers n={n_over}: P(terminates before 8192) mean {hit_over/max(1,n_over):.3f}')
# where is EOS competitive in normal answers? r_eos = z_eos / zmax along the answer
rs = []
for fn in sorted(os.listdir(d)):
    if fn.endswith('.npz'):
        a = np.load(os.path.join(d, fn)); rs.append((a['ze'] / a['topv'][:, 0])[:-1] if not a['trunc'] else (a['ze'] / a['topv'][:, 0]))
rs = np.concatenate(rs); print('r_eos over answer positions pct50/90/99/99.9', np.percentile(rs, [50, 90, 99, 99.9]).round(3))
