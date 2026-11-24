"""'Think-start guard': a </think>-row direction that lowers the close logit at the first generated positions only.
Ridge fit (2-fold CV by trace) of  hn.w = 1 at t < K, = 0 elsewhere (newline positions, natural-close positions weighted x20,
random other positions); then Delta e_close = -gamma * w  =>  Delta logit = -gamma * (hn.w).
usage: guard.py NPZ_DIR RESULT.json OUTPREFIX [K=1]"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, evaluate2
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model'))
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
D = load(sys.argv[1], sys.argv[2]); outp = sys.argv[3]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = np.random.default_rng(2)
def data(sel):
    X, y, wt = [], [], []
    for d in sel:
        T = len(d['lc']); h = d['hn'].astype(np.float32)
        nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
        pos = np.arange(min(K, T)); X.append(h[pos]); y.append(np.ones(len(pos))); wt.append(np.full(len(pos), 20.0))
        neg = np.where(nl & (np.arange(T) >= 16))[0]; X.append(h[neg]); y.append(np.zeros(len(neg))); wt.append(np.ones(len(neg)))
        oth = rng.choice(np.arange(16, T), size=min(200, max(0, T - 16)), replace=False) if T > 16 else np.array([], int)
        X.append(h[oth]); y.append(np.zeros(len(oth))); wt.append(np.ones(len(oth)))
        if d['closed']: X.append(h[-1:]); y.append(np.zeros(1)); wt.append(np.full(1, 20.0))
    return np.concatenate(X), np.concatenate(y), np.concatenate(wt)
def fit(sel, lam=10.0):
    X, y, wt = data(sel); Xw = X * np.sqrt(wt)[:, None]; yw = y * np.sqrt(wt)
    return np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ yw)
idx = rng.permutation(len(D)); folds = [idx[::2], idx[1::2]]
for lam in [1.0, 10.0, 100.0]:
    rows = []
    for k in (0, 1):
        w = fit([D[i] for i in folds[k]], lam); ev = [D[i] for i in folds[1 - k]]
        s0 = np.concatenate([d['hn'][:K].astype(np.float32) @ w for d in ev])
        sc = np.array([d['hn'][-1].astype(np.float32) @ w for d in ev if d['closed']])
        sn = []; so = []
        for d in ev:
            T = len(d['lc']); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]; h = d['hn'].astype(np.float32)
            sn.append(h[nl & (np.arange(T) >= 16)] @ w); so.append(h[16:] @ w)
        sn = np.concatenate(sn); so = np.concatenate(so)
        rows.append((np.percentile(s0, [5, 50]), np.percentile(sc, [50, 95]) if len(sc) else [np.nan] * 2,
                     np.percentile(sn, [50, 99, 99.9]), np.percentile(np.abs(so), [99, 99.9])))
    for r in rows:
        print(f'lam {lam:g}: heldout s at t<{K} (5%,50%) {np.round(r[0],3)} | natural-close (50,95%) {np.round(r[1],3)} | newline (50,99,99.9%) {np.round(r[2],3)} | |all t>=16| (99,99.9%) {np.round(r[3],3)}')
w = fit(D, 10.0); np.save(f'{outp}_guard_w.npy', w)
POP = {'ok': 143, 'never': 150, 'over': 27}
def st(d): return 'never' if not d['closed'] else ('over' if d['truncated'] else 'ok')
S = {k: [d for d in D if st(d) == k] for k in POP}
def pred(fn):
    out = {k: evaluate2(v, fn) for k, v in S.items() if v}
    return sum(POP[k] * out[k]['acc'] for k in out) / sum(POP[k] for k in out), sum(POP[k] * out[k]['trunc'] for k in out) / sum(POP[k] for k in out), {k: round(float(out[k]['acc']), 3) for k in out}
# in-sample-fit guard (w from all traces) used for the what-if; CV numbers above show how it transfers
for cg in [1.3, 1.35, 1.4, 1.5]:
    c = cg / 1.3
    for g in [0, 5, 10]:
        a, t, per = pred(lambda d, c=c, g=g: d['lc'] * c - g * (d['hn'].astype(np.float32) @ w))
        print(f'close_gain {cg:.2f} guard gamma {g:2d}: acc {100*a:.1f} trunc {t:.3f} per-stratum {per}')
# reference: guard as an oracle mask (no close allowed at t<K), to separate t=0 effects from mid-trace effects
for cg in [1.3, 1.35, 1.4, 1.5]:
    c = cg / 1.3
    def f(d, c=c):
        l = d['lc'] * c; l = l.copy(); l[:K] = d['lc'][:K] - 100; return l
    a, t, per = pred(f)
    print(f'close_gain {cg:.2f} ORACLE mask t<{K}: acc {100*a:.1f} trunc {t:.3f} per-stratum {per}')
