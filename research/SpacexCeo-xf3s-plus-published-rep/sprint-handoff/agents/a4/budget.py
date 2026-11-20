"""Is trace length (position) linearly decodable from the final-norm state hn at newline positions?
And what is the forced-close accuracy late in never-closed traces (value of a soft budget)?
usage: budget.py NPZ_DIR RESULT.json OUTPREFIX"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, p_close, value_curve
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model'))
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
D = load(sys.argv[1], sys.argv[2]); outp = sys.argv[3]
nv = [d for d in D if not d['closed']]
print('never-closed traces', len(nv))
for j, pos in enumerate([1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]):
    v = [int(d['pok'][list(d['ppos']).index(pos)]) for d in nv if pos in list(d['ppos'])]
    if v: print(f'  forced-close accuracy at {pos}: {np.mean(v):.3f} (n={len(v)})')
# ridge regression of position on hn at newline positions, 2-fold by trace
rng = np.random.default_rng(1); idx = rng.permutation(len(D)); folds = [idx[::2], idx[1::2]]
def xy(sel):
    X, y = [], []
    for d in sel:
        T = len(d['lc']); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
        if d['closed']: nl[-1] = False
        t = np.arange(T)[nl]; X.append(d['hn'][nl].astype(np.float32)); y.append(t)
    return np.concatenate(X), np.concatenate(y).astype(np.float32)
for lam in [1e2, 1e3, 1e4]:
    out = []
    for k in (0, 1):
        Xa, ya = xy([D[i] for i in folds[k]]); Xb, yb = xy([D[i] for i in folds[1 - k]])
        mx = Xa.mean(0); my = ya.mean()
        A = (Xa - mx).T @ (Xa - mx) + lam * np.eye(Xa.shape[1])
        w = np.linalg.solve(A, (Xa - mx).T @ (ya - my))
        pb = (Xb - mx) @ w + my
        r = np.corrcoef(pb, yb)[0, 1]
        bins = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 7000), (7000, 8200)]
        out.append((r, [(np.median(pb[(yb >= a) & (yb < b)]) if ((yb >= a) & (yb < b)).any() else np.nan) for a, b in bins],
                    [np.percentile(pb[(yb >= a) & (yb < b)], [10, 90]).round(0).tolist() if ((yb >= a) & (yb < b)).any() else None for a, b in bins]))
    print(f'lam {lam:g}: heldout corr {[round(o[0], 3) for o in out]}; median predicted pos per true bin (0-2k,2-4k,4-6k,6-7k,7k+): {[np.round(o[1], 0).tolist() for o in out]}')
    print('    10/90 pct per bin:', [o[2] for o in out])
# full fit for later use
X, y = xy(D); mx = X.mean(0); my = y.mean()
w = np.linalg.solve((X - mx).T @ (X - mx) + 1e3 * np.eye(X.shape[1]), (X - mx).T @ (y - my))
np.save(outp + '_posw.npy', w); np.save(outp + '_posmx.npy', mx); np.save(outp + '_posmy.npy', np.array([my]))
