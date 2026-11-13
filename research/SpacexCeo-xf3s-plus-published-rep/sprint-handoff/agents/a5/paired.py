"""Paired comparison of two proxy runs on shared (id, sample) seeds: acc, strict (normal-closed & correct), trunc, never/overflow."""
import json, sys, math, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
def load(name):
    R = {(r['id'], r['sample']): r for r in json.load(open(f'results/{name}.json'))['rows']}
    for l in open(f'results/{name}.texts.jsonl'):
        t = json.loads(l); r = R[(t['id'], t['sample'])]
        th = len(tok(t['text'].split('</think>')[0], add_special_tokens=False)['input_ids'])
        r['regime'] = 'never' if not r['closed'] else ('skip' if th < 50 else ('overflow' if r['truncated'] else 'normal'))
        r['strict'] = r['score'] if r['regime'] == 'normal' else 0.0
    return R
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
def m(R, f, ks): return 100 * sum(f(R[k]) for k in ks) / len(ks)
out = {}
for bench in ['ALL', 'math500', 'gsm8k', 'mmlu_pro', 'arc_c']:
    ks = [k for k in keys if bench == 'ALL' or a[k]['bench'] == bench]
    # paired SE over prompt ids (cluster samples of a prompt)
    ids = sorted({k[0] for k in ks}); diffs = []
    for i in ids:
        kk = [k for k in ks if k[0] == i]
        diffs.append(sum(b[k]['strict'] - a[k]['strict'] for k in kk) / len(kk))
    mu = sum(diffs) / len(diffs); se = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
    row = {'n': len(ks)}
    for nm, f in [('acc', lambda r: r['score']), ('strict', lambda r: r['strict']), ('trunc', lambda r: r['truncated']),
                  ('never', lambda r: r['regime'] == 'never'), ('overflow', lambda r: r['regime'] == 'overflow'), ('skip', lambda r: r['regime'] == 'skip')]:
        row[nm] = (round(m(a, f, ks), 2), round(m(b, f, ks), 2))
    nk = [k for k in ks if b[k]['regime'] == 'normal']; na = [k for k in ks if a[k]['regime'] == 'normal']
    row['acc_normal'] = (round(m(a, lambda r: r['score'], na), 1), round(m(b, lambda r: r['score'], nk), 1))
    row['d_strict'] = round(100 * mu, 2); row['se'] = round(100 * se, 2)
    out[bench] = row; print(bench, row)
