"""Server-side relations from lb.json (ground truth, no proxy): private vs truncation, tokens, validation benches."""
import json, numpy as np
from scipy import stats
d = json.load(open('/home/user/delta/lb.json'))
R = [r for r in d['rows'] if (r.get('result') or {}).get('state') == 'complete']
vid = [r['version_id'][:8] for r in R]
P = np.array([r['result']['test_score'] for r in R]); T = np.array([r['result']['stats']['truncation_rate'] for r in R])
TOK = np.array([r['result']['stats']['output_tokens'] for r in R]) / 500
V = np.array([r['result']['validation_score'] for r in R])
C = 1 - T
print('version  private trunc closed acc|closed tok/item val')
for i in range(len(R)): print(f'{vid[i]} {P[i]:6.1f} {T[i]:.3f} {C[i]:.3f} {100*P[i]/100/C[i]:6.1f} {TOK[i]:7.0f} {V[i]:.2f}')
def loo(x, y):
    e = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        b = np.polyfit(x[m], y[m], 1); e.append(y[i] - np.polyval(b, x[i]))
    return np.sqrt(np.mean(np.square(e)))
for name, mask in [('all 11', np.ones(len(R), bool)), ('excl. 0ffb5aef (think-forcing)', np.array([v != '0ffb5aef' for v in vid]))]:
    print('\n==', name)
    for lab, x in [('closed=1-trunc', C[mask]), ('tokens/item', TOK[mask]), ('validation', V[mask])]:
        y = P[mask]; b = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]; rho = stats.spearmanr(x, y)[0]
        print(f'{lab:16s} slope {b[0]:8.3f} icpt {b[1]:7.2f}  r {r:+.3f}  spearman {rho:+.3f}  LOO-RMSE {loo(x, y):.2f}')
print('\nper-validation-bench Spearman with private (n=11; val benches have 32-64 items each -> noisy):')
for b in sorted(R[0]['result']['validation_benchmarks']):
    x = np.array([r['result']['validation_benchmarks'][b] for r in R]); print(f'  {b:18s} {stats.spearmanr(x, P)[0]:+.2f}')
