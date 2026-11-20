"""Paired comparison of two per-item reports (likelihood_eval.py or gsm8k_sampled.py output).
Usage: python compare_items.py BASE.json EDIT.json [--resamples 10000 --seed 0]"""
import argparse, json, random, statistics
from math import comb
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('edit')
ap.add_argument('--resamples', type=int, default=10000); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
B, E = json.load(open(a.base))['groups'], json.load(open(a.edit))['groups']
rng = random.Random(a.seed); rows = {}
def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
for g in B:
    ids = sorted(B[g]['items']); assert ids == sorted(E[g]['items'])
    d = [E[g]['items'][i] - B[g]['items'][i] for i in ids]
    w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    rows[g] = {'n': len(d), 'base': B[g]['score'], 'edit': E[g]['score'], 'diff': 100 * statistics.mean(d),
               'wins': w, 'losses': l, 'sign_test_p': sign_p(w, l), 'd': d}
# Equal-weight mean over groups; bootstrap resamples items within each group jointly.
boots = []
for _ in range(a.resamples):
    boots.append(statistics.mean(100 * statistics.mean(rng.choices(r['d'], k=r['n'])) for r in rows.values()))
boots.sort()
W, L = sum(r['wins'] for r in rows.values()), sum(r['losses'] for r in rows.values())
overall = {'diff': statistics.mean(r['diff'] for r in rows.values()),
           'ci95': [boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]],
           'wins': W, 'losses': L, 'sign_test_p': sign_p(W, L)}
for r in rows.values(): r.pop('d')
print(json.dumps({'groups': rows, 'overall': overall}, indent=1))
