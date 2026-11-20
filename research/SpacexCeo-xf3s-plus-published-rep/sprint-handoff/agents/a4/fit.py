"""Closing statistics + conditional-closing direction fit + hazard-model evaluation (2-fold CV by trace).
usage: fit.py NPZ_DIR RESULT.json OUTPREFIX"""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, p_close, evaluate, value_curve, LANS
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model'))
D = load(sys.argv[1], sys.argv[2]); outp = sys.argv[3]
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
print('traces', len(D), 'closed', sum(d['closed'] for d in D), 'never-closed', sum(not d['closed'] for d in D))
# ---------- 1. closing statistics ----------
for d in D:
    T = len(d['lc'])
    d['p'] = p_close(d['lc'], d['top'])
    L = np.concatenate([d['lc'][:, None], d['top']], 1)
    d['rank'] = (d['top'] > d['lc'][:, None]).sum(1)
    nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]          # position t follows a token containing a newline
    d['nl'] = nl
    d['V'] = value_curve(d)
cl = [d for d in D if d['closed']]; tr = [d for d in D if not d['closed']]
pend = np.array([d['p'][-1] for d in cl]); rend = np.array([d['rank'][-1] for d in cl])
print(f'closed traces: p(close) at natural close median {np.median(pend):.3f}, rank median {np.median(rend)}, in top-20 {np.mean(rend < 20):.2f}')
print(f'closed traces: sum of p over thinking (excl. close) median {np.median([d["p"][:-1].sum() for d in cl]):.3f}')
print(f'never-closed: sum p over 8192 median {np.median([d["p"].sum() for d in tr]):.3f}, mean {np.mean([d["p"].sum() for d in tr]):.3f}')
print(f'never-closed: frac positions with close in top-20: {np.mean(np.concatenate([d["rank"] < 20 for d in tr])):.4f};  closed traces: {np.mean(np.concatenate([d["rank"][:-1] < 20 for d in cl])):.4f}')
print(f'p(close) at newline positions vs others (never-closed): {np.mean(np.concatenate([d["p"][d["nl"]] for d in tr])):.2e} vs {np.mean(np.concatenate([d["p"][~d["nl"]] for d in tr])):.2e}')
kn = [d for d in tr if d['V'].max() > 0]
print(f'never-closed with a correct forced-close probe somewhere: {len(kn)}/{len(tr)}')
for d in kn:
    i = int(np.argmax(d['V'] > 0));
    print(f'   {d["key"]:8s} {d["bench"]:8s} first-correct probe at {i}, probes {"".join(str(int(x)) for x in d["pok"])}, '
          f'sum p after = {d["p"][i:].sum():.3f}, max p after {d["p"][i:].max():.3f}, median rank at newlines after {np.median(d["rank"][i:][d["nl"][i:]]):.0f}')
# forced-close probe at the natural close of closed traces vs actual score (validates the probe as a value estimate)
agree = [(int(d['pok'][-1]), int(d['score'] > 0)) for d in cl]
print('probe@close vs actual score (closed traces): ', {k: agree.count(k) for k in set(agree)})
# ---------- 2. directions ----------
def stack(sel, which):
    H = []
    for d in sel:
        m = which(d)
        if m.any(): H.append(d['hn'][m].astype(np.float32))
    return np.concatenate(H) if H else np.zeros((0, 1024), np.float32)
def directions(sel):
    c_ = [d for d in sel if d['closed']]
    end = stack(c_, lambda d: np.arange(len(d['lc'])) == len(d['lc']) - 1)
    mid = stack(sel, lambda d: d['nl'] & (np.arange(len(d['lc'])) < len(d['lc']) - 64) & (d['V'] == 0))
    known = stack([d for d in sel if not d['closed']], lambda d: d['nl'] & (d['V'] > 0))
    mid_all = stack(sel, lambda d: d['nl'] & (np.arange(len(d['lc'])) < len(d['lc']) - 64))
    out = {}
    if len(end) and len(mid): out['close'] = (end.mean(0), mid.mean(0))
    if len(known) and len(mid): out['known'] = (known.mean(0), mid.mean(0))
    return out, mid_all.mean(0)
rng = np.random.default_rng(0); idx = rng.permutation(len(D)); folds = [idx[::2], idx[1::2]]
def edit_fn(u, m0, beta, c=1.0, nl_only=False):
    def f(d):
        s = (d['hn'].astype(np.float32) - m0) @ u
        return d['lc'] * c + beta * s
    return f
res = {}
full, _ = directions(D)
for name, (a, b) in full.items():
    dvec = a - b; u = dvec / dvec.dot(dvec); np.save(f'{outp}_{name}_u.npy', u); np.save(f'{outp}_{name}_m0.npy', b)
    print(f'direction {name}: |d|={np.linalg.norm(dvec):.2f}, cos with e_close-row proxy n/a')
print('baseline (parent) on all:', {k: round(v, 4) for k, v in evaluate(D, lambda d: d['lc']).items()})
for name in full:
    for beta in [1, 2, 3, 4, 6]:
        tot = []
        for k in (0, 1):
            fit_ = [D[i] for i in folds[k]]; ev = [D[i] for i in folds[1 - k]]
            dd, _ = directions(fit_)
            if name not in dd: continue
            a, b = dd[name]; dvec = a - b; u = dvec / dvec.dot(dvec)
            tot.append(evaluate(ev, edit_fn(u, b, beta)))
        r = {k: round(np.mean([t[k] for t in tot]), 4) for k in tot[0]}
        print(f'CV {name} beta={beta}:', r)
for c in [1.05, 1.1, 1.15]:
    print(f'row x{c} (close_gain {1.3*c:.3f}):', {k: round(v, 4) for k, v in evaluate(D, lambda d, c=c: d['lc'] * c).items()})
# ---------- 3. separation of classes along fitted directions (out-of-fold) ----------
def classes(sel):
    K, U, Pp, Ee = [], [], [], []
    for d in sel:
        T = len(d['lc']); nl = d['nl'].copy()
        if d['closed']:
            nl[-1] = False; Pp.append(d['hn'][nl & (np.arange(T) < T - 32)]); Ee.append(d['hn'][-1:])
        else:
            K.append(d['hn'][nl & (d['V'] > 0)]); U.append(d['hn'][nl & (d['V'] == 0)])
    f = lambda L: np.concatenate(L).astype(np.float32) if L else np.zeros((0, 1024), np.float32)
    return f(K), f(U), f(Pp), f(Ee)
def lda(pos, neg, lam=0.1):
    X = np.concatenate([pos - pos.mean(0), neg - neg.mean(0)])
    S = X.T @ X / len(X); S = (1 - lam) * S + lam * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w = np.linalg.solve(S, pos.mean(0) - neg.mean(0)); return w
for k in (0, 1):
    fit_ = [D[i] for i in folds[k]]; ev = [D[i] for i in folds[1 - k]]
    K, U, Pp, Ee = classes(fit_); K2, U2, P2, E2 = classes(ev)
    if len(K) == 0 or len(K2) == 0: continue
    neg = np.concatenate([U, Pp])
    for name, w in [('meandiff K-(U+P)', K.mean(0) - neg.mean(0)), ('lda K-(U+P)', lda(K, neg)), ('lda K-(U+P) lam.5', lda(K, neg, .5))]:
        m0 = neg.mean(0); sc = (K.mean(0) - m0) @ w
        s = lambda X: ((X - m0) @ w) / sc                         # 0 = negative mean, 1 = positive mean (fit fold)
        qs = lambda X: ' '.join(f'{v:5.2f}' for v in np.percentile(s(X), [10, 50, 90])) if len(X) else 'n/a'
        print(f'fold{k} {name:18s} heldout s-quantiles 10/50/90: known-loop {qs(K2)} | unknown-loop {qs(U2)} | pre-close {qs(P2)} | close-pos {qs(E2)}')
