"""Premature-close suppression on b8af1dbc: direction separating hn at the close position of OVERFLOW rows (closed early,
answer phase hits 8192) from hn at the close position of NORMAL rows. Edit: e_close -= beta * u,  u = d/|d|^2 centred so
that Delta logit = -beta * s,  s = (hn - m_normal).d/|d|^2  (0 at normal-close mean, 1 at overflow-close mean).
The centring constant is not realisable exactly (no constant direction), so we also report the UNcentred variant
Delta logit = -beta * hn.d/|d|^2 + const_offset_from_close_gain (row scale) -- see output.
usage: prem.py NPZ_DIR RESULT.json OUTPREFIX"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, p_close
D = load(sys.argv[1], sys.argv[2]); outp = sys.argv[3]
cl = [d for d in D if d['closed'] and len(d['lc']) > 50]
ov = [d for d in cl if d['truncated']]; nm = [d for d in cl if not d['truncated']]
print('closed traces', len(cl), 'overflow', len(ov), 'normal', len(nm))
H = lambda S: np.array([d['hn'][-1].astype(np.float32) for d in S])
def pk(d, delta):  # keep prob of the natural close after adding delta to its logit
    p0 = p_close(d['lc'][-1:], d['top'][-1:])[0]; p1 = p_close(d['lc'][-1:] + delta, d['top'][-1:])[0]
    return min(1.0, p1 / max(p0, 1e-12))
rng = np.random.default_rng(5)
res = {b: [[], [], []] for b in [1, 2, 3, 4, 6]}
aucs = []
for rep in range(20):   # repeated 2-fold CV over traces
    io = rng.permutation(len(ov)); inn = rng.permutation(len(nm))
    for k in (0, 1):
        tro = [ov[i] for i in io[k::2]]; tes_o = [ov[i] for i in io[1 - k::2]]
        trn = [nm[i] for i in inn[k::2]]; tes_n = [nm[i] for i in inn[1 - k::2]]
        mo, mn = H(tro).mean(0), H(trn).mean(0); dv = mo - mn; u = dv / dv.dot(dv)
        so = (H(tes_o) - mn) @ u; sn = (H(tes_n) - mn) @ u
        aucs.append(np.mean(so[:, None] > sn[None, :]))
        for b in res:
            res[b][0] += [pk(d, -b * s) for d, s in zip(tes_o, so)]
            res[b][1] += [pk(d, -b * s) for d, s in zip(tes_n, sn)]
            res[b][2] += [d['score'] for d in tes_n]
print(f'held-out AUC overflow-vs-normal close states: mean {np.mean(aucs):.3f} (sd {np.std(aucs):.3f})')
for b, (ko, kn, sc) in res.items():
    ko, kn, sc = map(np.array, (ko, kn, sc))
    print(f'beta {b}: P(close kept) overflow {ko.mean():.3f} | normal {kn.mean():.3f} (correct normals {kn[sc > 0].mean():.3f})')
# position-0 and newline side effects of the full-data direction
mo, mn = H(ov).mean(0), H(nm).mean(0); dv = mo - mn; u = dv / dv.dot(dv)
np.save(outp + '_u.npy', u); np.save(outp + '_mn.npy', mn)
off = mn @ u
print(f'|d| {np.linalg.norm(dv):.2f}; centring offset m_normal.u = {off:.3f}')
allh = np.concatenate([d['hn'][::5].astype(np.float32) for d in D]); s_all = (allh - mn) @ u
print('s over all positions (5,50,95%):', np.round(np.percentile(s_all, [5, 50, 95]), 3), ' s at t=0 (median):', round(float(np.median([(d['hn'][0].astype(np.float32) - mn) @ u for d in D])), 3))
# uncentred variant: Delta logit = -beta*(hn.u); its value at normal closes = -beta*(off + s)
print('uncentred hn.u at normal closes (5,50,95%):', np.round(np.percentile(H(nm) @ u, [5, 50, 95]), 3), ' at overflow closes:', np.round(np.percentile(H(ov) @ u, [5, 50, 95]), 3))
# ---------- realisable edit: e_close <- c*e_close - beta*u  (uncentred), CV ----------
print('--- realisable edit  Delta = (c-1)*lc - beta*hn.u ; CV keep-probs + extra early-close hazard in normal traces ---')
for beta, c in [(0, 0.97), (0, 0.95), (0, 0.92), (0, 0.90), (2, 0.96), (2, 0.97), (3, 0.94), (3, 0.95), (4, 0.93), (4, 0.92)]:
    ko, kn, kc, ex = [], [], [], []
    for rep in range(10):
        io = rng.permutation(len(ov)); inn = rng.permutation(len(nm))
        for k in (0, 1):
            tro = [ov[i] for i in io[k::2]]; tes_o = [ov[i] for i in io[1 - k::2]]
            trn = [nm[i] for i in inn[k::2]]; tes_n = [nm[i] for i in inn[1 - k::2]]
            dv = H(tro).mean(0) - H(trn).mean(0); uu = dv / dv.dot(dv)
            for d in tes_o + tes_n:
                h = d['hn'].astype(np.float32); delta = (c - 1) * d['lc'] - beta * (h @ uu)
                keep = pk(d, delta[-1])
                (ko if d['truncated'] else kn).append(keep)
                if not d['truncated']:
                    kc.append(keep) if d['score'] > 0 else None
                    T = len(d['lc']) - 1
                    p0 = p_close(d['lc'][:T], d['top'][:T]); p1 = p_close(d['lc'][:T] + delta[:T], d['top'][:T])
                    ex.append(1 - np.prod(1 - np.clip((p1 - p0) / np.maximum(1e-12, 1 - p0), 0, 1)))
    print(f'beta {beta} c {c}: keep overflow {np.mean(ko):.3f} | keep normal {np.mean(kn):.3f} (correct {np.mean(kc):.3f}) | P(extra earlier close) in normal traces {np.mean(ex):.3f}')
