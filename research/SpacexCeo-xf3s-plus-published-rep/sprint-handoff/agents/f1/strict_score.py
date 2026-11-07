"""a6: break proxy accuracy into regimes: skip-thinking (<50 thinking tokens), closed-overflow (closed but hit max_tokens),
normal closed, never closed; 'strict' acc counts only normal closed generations (advisory heuristic for private)."""
import json, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
def regimes(name):
    R = {(r['id'], r['sample']): r for r in json.load(open(f'results/{name}.json'))['rows']}
    g = collections.defaultdict(list)
    for l in open(f'results/{name}.texts.jsonl'):
        t = json.loads(l); r = R[(t['id'], t['sample'])]
        if not r['closed']: k = 'never'
        elif len(tok(t['text'].split('</think>')[0], add_special_tokens=False)['input_ids']) < 50: k = 'skip'
        elif r['truncated']: k = 'overflow'
        else: k = 'normal'
        g[k].append((r['score'], t['bench']))
    n = sum(len(v) for v in g.values())
    out = {k: (len(v), round(100 * sum(x for x, _ in v) / max(1, len(v)), 1)) for k, v in sorted(g.items())}
    strict = 100 * sum(x for x, _ in g['normal']) / n
    return out, round(strict, 2)
if __name__ == '__main__':
    for name in sys.argv[1:]:
        o, s = regimes(name); print(name, 'regime (n, acc%):', o, 'strict_acc', s)
