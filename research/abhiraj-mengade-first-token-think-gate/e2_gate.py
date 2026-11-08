"""E2: run a candidate in memory under the server's thinking-on prompt.

Usage: e2_gate.py NAME GAIN DELTA.npy [--filler S] [--cap N] [--limit N] [--parent PATH]
"""
import argparse
import json
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, 'analysis')
import harness as H

a = argparse.ArgumentParser()
a.add_argument('name')
a.add_argument('gain', type=float)
a.add_argument('delta')
a.add_argument('--filler', type=float, default=1.0)
a.add_argument('--cap', type=int, default=2048)
a.add_argument('--limit', type=int, default=0)
a.add_argument('--batch', type=int, default=12)
a.add_argument('--parent', default='experiment/model')
a.add_argument('--bare', action='store_true', help='remove the answer-format instruction from each prompt')
a.add_argument('--skip', type=int, default=0, help='skip the first N items of each benchmark')
args = a.parse_args()

SUFFIXES = (r' Please reason step by step and put your final answer in \boxed{}.',
            r'Answer with the letter of the correct option in \boxed{}.')
bank = json.load(open('analysis/items.json'))
items = [dict(x) for v in bank.values() for x in v[args.skip:args.limit or None]]
if args.bare:
    for x in items:
        for s in SUFFIXES:
            x['prompt'] = x['prompt'].replace(s, '')
        x['prompt'] = x['prompt'].strip()
    assert not any('boxed' in x['prompt'] and x['scorer'] == 'choice' for x in items)
model, tok = H.load_model(args.parent)
emb, norm = H.params(model)
norm.weight = norm.weight * args.gain
W = emb.weight
if args.delta != 'none':
    d = mx.array(np.load(args.delta)).astype(W.dtype)
    W[H.CLOSE] = W[H.CLOSE] + d
FILLER = [3655, 11158, 13428, 13784, 77264, 84485, 85152]  # ' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm'
if args.filler != 1.0:
    W[mx.array(FILLER)] = W[mx.array(FILLER)] * args.filler
emb.weight = W
mx.eval(emb.weight, norm.weight)
rows = H.run(model, tok, items, thinking=True, max_tokens=args.cap, batch=args.batch, log=f'analysis/e2_{args.name}.jsonl')
s = H.summary(rows)
closed0 = sum(r['close_at'] is not None and r['close_at'] <= 2 for r in rows) / len(rows)
print(args.name, 'close_by_t2', round(closed0, 3), json.dumps(s), flush=True)
