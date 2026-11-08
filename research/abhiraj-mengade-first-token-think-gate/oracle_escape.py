"""Upper bound on escaping answer-stage loops: cut each truncated answer at its first repeated line, inject a
closing phrase, and score the completion. An oracle: it knows where to intervene. Decides if a stop gate is worth fitting.

Usage: oracle_escape.py LOG GAIN DELTA.npy FILLER
"""
import json
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, 'analysis')
import harness as H

log, gain, delta, filler = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
PHRASES = ['\n\n**Final Answer**\n', '\n\nTherefore, the final answer is $\\boxed{']
FILLER = [3655, 11158, 13428, 13784, 77264, 84485, 85152]


def cut(text):
    """Text up to the second occurrence of the first repeated long line, else 60% through."""
    seen = set()
    pos = 0
    for line in text.split('\n'):
        s = line.strip()
        if len(s) >= 20:
            if s in seen:
                return text[:pos], True
            seen.add(s)
        pos += len(line) + 1
    return text[:int(.6 * len(text))], False


bank = json.load(open('analysis/items.json'))
ex = {x['id']: x for v in bank.values() for x in v}
rows = [json.loads(l) for l in open(log)]
rows = [r for r in rows if r['truncated']]
model, tok = H.load_model()
emb, norm = H.params(model)
norm.weight = norm.weight * gain
W = emb.weight
W[H.CLOSE] = W[H.CLOSE] + mx.array(np.load(delta)).astype(W.dtype)
if filler != 1.0:
    W[mx.array(FILLER)] = W[mx.array(FILLER)] * filler
emb.weight = W
mx.eval(emb.weight, norm.weight)

for pi, phrase in enumerate(PHRASES):
    out = []
    for r in rows:
        body, is_loop = cut(r['text'])
        it = ex[r['id']]
        ids = H.render(tok, it['prompt'], True)
        pre = tok.encode(body + phrase, add_special_tokens=False)
        res = H.run(model, tok, [dict(it)], thinking=True, max_tokens=200, batch=1, prefix=pre)[0]
        text = ('\\boxed{' if phrase.endswith('{') else '') + res['text']
        score = H.score_answer(it, '</think>' + text)
        out.append(dict(id=r['id'], benchmark=r['benchmark'], loop=is_loop, score=score, cut_chars=len(body), text=text[:300]))
    k = len(out)
    print(f'phrase {pi} {phrase!r}: n={k} right={sum(x["score"] for x in out):.0f} loop_cut={sum(x["loop"] for x in out)} '
          f'right_on_loop={sum(x["score"] for x in out if x["loop"]):.0f}', flush=True)
    by = {}
    for x in out:
        by.setdefault(x['benchmark'], []).append(x['score'])
    print('  ', {b: (len(v), sum(v)) for b, v in by.items()}, flush=True)
    json.dump(out, open(f'analysis/oracle_{pi}.json', 'w'), indent=0)
print('base rows', len(rows), 'base score', sum(r['score'] for r in rows))
