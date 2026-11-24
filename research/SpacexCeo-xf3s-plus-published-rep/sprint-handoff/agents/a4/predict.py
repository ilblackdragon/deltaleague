"""Stratum-reweighted hazard-model predictions (proxy, 320 prompts) for close-row edits on the c_new parent.
Strata (c_new proxy counts): closed-ok 143, never-closed 150, overrun (closed but truncated) 27.
usage: predict.py NPZ_DIR RESULT.json"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, evaluate2
D = load(sys.argv[1], sys.argv[2])
POP = {'ok': 143, 'never': 150, 'over': 27}
def st(d): return 'never' if not d['closed'] else ('over' if d['truncated'] else 'ok')
S = {k: [d for d in D if st(d) == k] for k in POP}
print('sample sizes', {k: len(v) for k, v in S.items()})
def pred(fn, **kw):
    out = {}
    for k, v in S.items():
        if v: out[k] = evaluate2(v, fn, **kw)
    acc = sum(POP[k] * out[k]['acc'] for k in out) / sum(POP[k] for k in out)
    tr = sum(POP[k] * out[k]['trunc'] for k in out) / sum(POP[k] for k in out)
    return acc, tr, {k: round(float(out[k]['acc']), 3) for k in out}
a0, t0, _ = pred(lambda d: d['lc'])
print(f'parent (as sampled): acc {100*a0:.1f} trunc {t0:.3f}   [real c_new proxy: 33.4 / 0.553]')
for cg in [1.15, 1.2, 1.25, 1.35, 1.4, 1.45, 1.5]:
    c = cg / 1.3
    for rho in ([0.85, 1.0] if c < 1 else [0.85]):
        a, t, per = pred(lambda d, c=c: d['lc'] * c, rho=rho)
        print(f'close_gain {cg:.2f} (row x{c:.3f} vs parent){" rho="+str(rho) if c<1 else ""}: acc {100*a:.1f} trunc {t:.3f}  per-stratum acc {per}')
