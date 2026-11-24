"""a6: evaluate counterfactual (s, c, g) specs on stored teacher-forced logits (from tf_hazard.py). First-order, advisory.
Logits: g*z0 (ordinary) | g*s*z0 (7 scaled hedge rows) | g*z0 (6 Maybe/Actually/Alternatively rows) | g*c*z0 (</think>),
minus presence penalty 1.5 on already-generated tokens, then top_k 20, top_p 0.95, T=1 (vLLM order).
usage: eval_hazard.py NPZ [OUT.json]"""
import sys, json, numpy as np
D = np.load(sys.argv[1]); zt = D['zt'].astype(np.float32); st = D['st']; zs = D['zs']; ss = D['ss']
think = D['think']; nxt = D['nxt_close']; closed = D['closed'].astype(bool); pos = D['pos']; trace = D['trace']
Z = np.concatenate([zt, zs], 1); S = np.concatenate([st, ss], 1).astype(np.float32); K = zt.shape[1]
iH7 = np.arange(K, K + 7); iHO = np.arange(K + 7, K + 13); iC = K + 13
# last 30 think positions before a natural close are 'closing zone'
zone = np.zeros(len(pos), bool)
for t in np.unique(trace):
    m = trace == t; idx = np.where(m & nxt)[0]
    if len(idx): zone[max(idx[0] - 30, np.where(m)[0][0]): idx[0] + 1] = True
def probs(s, c, g, sel):
    """returns (idx[n,20], p[n,20]): the top-20 candidates and their final sampling probabilities (0 if cut by top_p)"""
    mult = np.ones(Z.shape[1], np.float32); mult[iH7] = s; mult[iC] = c
    L = g * Z[sel] * mult - 1.5 * S[sel]
    idx = np.argpartition(-L, 20, axis=1)[:, :20]; Lk = np.take_along_axis(L, idx, 1)
    o = np.argsort(-Lk, 1); idx = np.take_along_axis(idx, o, 1); Ls = np.take_along_axis(Lk, o, 1)
    p = np.exp(Ls - Ls[:, :1]); p /= p.sum(1, keepdims=True)
    cum = np.cumsum(p, 1); keep = (cum - p) < 0.95
    p = np.where(keep, p, 0); p /= p.sum(1, keepdims=True)
    return idx, p
def mass(idx, p, cols):
    return (p * np.isin(idx, cols)).sum(1)
SUB = (think & ((pos % 4 == 0) | (pos < 50))) | nxt        # evaluated positions: every 4th think position + all close/early ones
RATE = (pos % 4 == 0) & think                                   # uniform subsample used for rate metrics
def metrics(s, c, g):
    idx, P = probs(s, c, g, SUB)
    ph7 = mass(idx, P, iH7); pho = mass(idx, P, iHO); pc = mass(idx, P, [iC])
    cl = closed[SUB]; zn = zone[SUB]; nx = nxt[SUB]; ps = pos[SUB]; rt = RATE[SUB]
    ent = -(np.where(P > 0, P * np.log(np.maximum(P, 1e-30)), 0)).sum(1)
    f = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    r = {'s': s, 'c': c, 'g': g,
         'hedge7_per1k': 1000 * ph7[rt].mean(), 'hedgeO_per1k': 1000 * pho[rt].mean(),
         'close_at_natural_end': pc[nx].mean() if nx.any() else None,
         'close_p_pos0': pc[ps == 0].mean(), 'close_haz_first50_sum': pc[ps < 50].sum() / max(1, (ps == 0).sum()),
         'close_haz_early_per1k': 1000 * pc[rt & cl & ~zn].mean(),              # premature-close hazard in traces that closed
         'close_haz_trunc_late_per1k': 1000 * pc[rt & ~cl & (ps >= 2000)].mean(),  # rescue hazard in traces that never closed
         'entropy': ent[rt].mean()}
    return {k: (f(v) if k not in 'scg' else v) for k, v in r.items()}
if __name__ == '__main__':
    R = []
    L = (0.5, 1.3, 1.4)
    for s in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
        for g in [1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]:
            R.append(metrics(s, L[1], g))
    for c in [1.0, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6]:
        for g in [1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]:
            R.append(metrics(L[0], c, g))
    for s, c, g in [(0.45, 1.4, 1.4), (0.45, 1.3, 1.4), (0.5, 1.4, 1.35), (0.55, 1.3, 1.3), (0.45, 1.3, 1.5), (0.4, 1.0, 1.0), (1.0, 1.5, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.25), (1.0, 1.0, 1.667)]:
        R.append(metrics(s, c, g))
    json.dump(R, open(sys.argv[2] if len(sys.argv) > 2 else 'hazard.json', 'w'), indent=0)
    for r in R: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
