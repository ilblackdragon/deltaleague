"""CV evaluation of a soft-budget close edit:  lc' = lc + a * (pos_hat(hn) - tau) / 1000,
pos_hat = ridge readout of position from hn (fit on the other fold). Implementable as e_close += (a/1000) w plus a constant
realised through the mean-embedding direction (needs hn.mu/|mu|^2 ~ const; its spread is printed).
usage: budget2.py NPZ_DIR RESULT.json PARENT_MODEL_DIR"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, evaluate2
from transformers import AutoTokenizer
from safetensors import safe_open
tok = AutoTokenizer.from_pretrained(sys.argv[3])
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
E = safe_open(os.path.join(sys.argv[3], 'model.safetensors'), 'pt').get_tensor('model.language_model.embed_tokens.weight').float().numpy()
mu = E.mean(0); umu = mu / mu.dot(mu)
D = load(sys.argv[1], sys.argv[2])
allh = np.concatenate([d['hn'][::7].astype(np.float32) for d in D]); cm = allh @ umu
print(f'traces {len(D)}; hn.mu/|mu|^2: mean {cm.mean():.3f} std {cm.std():.3f} (5/95%: {np.percentile(cm,5):.3f} {np.percentile(cm,95):.3f})')
def xy(sel, lam=1e3):
    X, y = [], []
    for d in sel:
        T = len(d['lc']); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
        if d['closed']: nl[-1] = False
        X.append(d['hn'][nl].astype(np.float32)); y.append(np.arange(T)[nl])
    X = np.concatenate(X); y = np.concatenate(y).astype(np.float32)
    mx = X.mean(0); my = y.mean()
    w = np.linalg.solve((X - mx).T @ (X - mx) + lam * np.eye(X.shape[1]), (X - mx).T @ (y - my))
    return w, mx, my
rng = np.random.default_rng(1); idx = rng.permutation(len(D)); folds = [idx[::2], idx[1::2]]
W = [xy([D[i] for i in folds[k]]) for k in (0, 1)]
def fn(w, mx, my, a, tau, use_mu):
    def f(d):
        h = d['hn'].astype(np.float32)
        if use_mu:   # constant realised through mu: c(h) * K with K chosen so that the mean of c(h) gives the intended constant
            K = (my - mx @ w - tau) / cm.mean()
            return d['lc'] + (a / 1000) * (h @ w + K * (h @ umu))
        return d['lc'] + (a / 1000) * ((h - mx) @ w + my - tau)
    return f
base = evaluate2(D, lambda d: d['lc'])
print('parent', {k: round(float(v), 4) for k, v in base.items()})
for rho in [0.85]:
    for tau in [5000, 6000, 6500, 7000]:
        for a in [0.5, 1, 2, 3]:
            for use_mu in [False, True]:
                R = []
                for k in (0, 1):
                    w, mx, my = W[k]; ev = [D[i] for i in folds[1 - k]]
                    R.append(evaluate2(ev, fn(w, mx, my, a, tau, use_mu), rho=rho))
                r = {kk: round(float(np.mean([x[kk] for x in R])), 4) for kk in R[0]}
                print(f'tau {tau} a {a} mu {int(use_mu)}:', r)
