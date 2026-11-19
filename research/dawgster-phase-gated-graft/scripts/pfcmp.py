"""Paired comparison of pubfast results. usage: python3 scripts/pfcmp.py BASE CAND [seeds=20261104,7]
Score = mean of group means (like the validation aggregate), per-item averaged over common seeds."""
import json, random, sys

base, cand = sys.argv[1], sys.argv[2]
seeds = (sys.argv[3] if len(sys.argv) > 3 else '20261104,7').split(',')


def load(n):
    out, tr = {}, {}
    for s in seeds:
        try:
            d = json.load(open(f'runs/pf_{n}_s{s}.scores.json'))
        except FileNotFoundError:
            continue
        for g, it in d['items'].items():
            for k, v in it.items():
                out.setdefault((g, k), []).append(v)
        for l in open(f'runs/pf_{n}_s{s}.jsonl'):
            r = json.loads(l)
            tr.setdefault(r['group'], []).append('</think>' not in r['text'] or r['finish'] == 'length')
    return out, tr


A, TA = load(base)
B, TB = load(cand)
keys = [k for k in A if k in B and len(A[k]) == len(B[k])]
groups = sorted({g for g, _ in keys})
nseeds = min(len(A[k]) for k in keys)
by = {g: [(sum(A[k]) / len(A[k]), sum(B[k]) / len(B[k])) for k in keys if k[0] == g] for g in groups}


def agg(by):
    a = sum(sum(x for x, _ in v) / len(v) for v in by.values()) / len(by)
    b = sum(sum(y for _, y in v) / len(v) for v in by.values()) / len(by)
    return 100 * a, 100 * b


a, b = agg(by)
rng = random.Random(0)
boots = []
for _ in range(2000):
    bb = {g: [v[rng.randrange(len(v))] for _ in v] for g, v in by.items()}
    x, y = agg(bb)
    boots.append(y - x)
boots.sort()
print(f'{base} {a:.2f} -> {cand} {b:.2f}  diff {b - a:+.2f}  95%CI [{boots[50]:+.2f},{boots[1950]:+.2f}]  seeds={nseeds}')
for g in groups:
    v = by[g]
    x = 100 * sum(p for p, _ in v) / len(v); y = 100 * sum(q for _, q in v) / len(v)
    ta = 100 * sum(TA[g]) / len(TA[g]); tb = 100 * sum(TB[g]) / len(TB[g])
    print(f'  {g:10s} {x:5.1f} -> {y:5.1f} ({y - x:+5.1f})   len-cap {ta:4.1f}% -> {tb:4.1f}%')
