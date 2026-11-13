"""Macro fit on the public leaderboard (ground truth): private = C * a(C) where C = 1 - trunc, a = accuracy given closed.
Plus the new leader point from the brief (31.40 private; its trunc is not in lb.json -> use proxy-implied value when known)."""
import json, numpy as np
d = json.load(open('/home/user/delta/lb.json'))
R = [r for r in d['rows'] if (r.get('result') or {}).get('state') == 'complete']
rows = []
for r in R:
    res = r['result']; st = res['stats']
    rows.append((r['version_id'][:8], res['test_score'], 1 - st['truncation_rate'], st['output_tokens'] / 500, res['validation_score'], r['name']))
rows.sort(key=lambda x: x[2])
print('version  private  closed  acc|closed  tok/item  val   name')
for v, p, C, tk, val, name in rows: print(f'{v} {p:6.1f} {C:.3f} {100*p/100/C:7.1f} {tk:8.0f} {val:5.1f}  {name[:60]}')
# fit a(C) excluding the think-forcing point (0ffb5aef) and the rep edits (different mechanism, lower val)
X = np.array([(C, p) for v, p, C, *_ in rows if v not in ('0ffb5aef',)])
C, P = X[:, 0], X[:, 1]
# model 1: cumulative accuracy curve: score(C) = A*(1-(1-C)^k)  (items ordered by closability; marginal acc = A*k*(1-C)^(k-1))
from scipy.optimize import curve_fit
f1 = lambda C, A, k: A * (1 - (1 - C) ** k)
pa, _ = curve_fit(f1, C, P, p0=[45, 2])
print('\nfit score = A(1-(1-C)^k):', pa.round(3), 'RMSE', np.sqrt(np.mean((P - f1(C, *pa)) ** 2)).round(2))
for c in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]: print(f'  C={c:.2f} -> score {f1(c,*pa):.1f}  marginal acc {pa[0]*pa[1]*(1-c)**(pa[1]-1):.2f}')
# model 2: linear in C
b = np.polyfit(C, P, 1); print('linear fit slope/icpt', b.round(2), 'RMSE', np.sqrt(np.mean((P - np.polyval(b, C)) ** 2)).round(2))
# where does a9340e14 (31.4) sit? solve C from each model
from scipy.optimize import brentq
print('new leader 31.4 implies C =', round(brentq(lambda c: f1(c, *pa) - 31.4, 0.2, 0.99), 3), '(curve) /', round((31.4 - b[1]) / b[0], 3), '(linear)')
print('target 34 implies C =', round(brentq(lambda c: f1(c, *pa) - 34, 0.2, 0.99), 3), ' 36 ->', round(brentq(lambda c: f1(c, *pa) - 36, 0.2, 0.99), 3))
