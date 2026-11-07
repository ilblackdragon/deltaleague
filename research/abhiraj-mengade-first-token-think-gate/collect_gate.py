"""Collect final-norm hidden states for fitting the </think> row, on prompts disjoint from items.json.

Positives: the position right after `<think>\n`, across several prompt styles, because the private prompt
format is unknown and a gate that only fires on our format would be worthless.
Negatives: every position of no-think answers, sampled under server decoding.
"""
import json
import random
import sys

import mlx.core as mx
import numpy as np
from datasets import load_dataset

sys.path.insert(0, 'analysis')
import harness as H

N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 90
LET = 'ABCDEFGHIJ'
rng = random.Random(7)

used = {x['id'] for v in json.load(open('analysis/items.json')).values() for x in v}
val = json.load(open('experiment/validation.json'))
val_math = {i for s in val['text']['math500']['selection'].values() for i in s.get('indices', [])}

raw = []
for d in load_dataset('openai/gsm8k', 'main', split='train').shuffle(seed=1).select(range(400)):
    raw.append(('math', d['question']))
m = load_dataset('HuggingFaceH4/MATH-500', split='test')
for i in range(len(m)):
    if f'math500-{i}' not in used and i not in val_math:
        raw.append(('math', m[i]['problem']))
for d in load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train').shuffle(seed=1).select(range(300)):
    raw.append(('mc', d['question'], d['choices']['text']))
for d in load_dataset('cais/mmlu', 'all', split='validation').shuffle(seed=1).select(range(300)):
    raw.append(('mc', d['question'], d['choices']))
for d in load_dataset('tau/commonsense_qa', split='train').shuffle(seed=1).select(range(300)):
    raw.append(('mc', d['question'], d['choices']['text']))
for d in load_dataset('google/IFEval', split='train').shuffle(seed=1).select(range(200)):
    raw.append(('free', d['prompt']))
rng.shuffle(raw)

MATH_STYLES = ['{q}', r'{q} Please reason step by step and put your final answer in \boxed{{}}.',
               r'{q}' '\n\n' r'Put your final answer within \boxed{{}}.', r'Solve the following problem. {q}',
               'Question: {q}\nAnswer:', r'Problem: {q}' '\n' r'Give the final answer in \boxed{{}}.',
               'Solve this and end with the numeric result only.\n\n{q}']
MC_STYLES = [r'{q}' '\n{o}\n' r'Answer with the letter of the correct option in \boxed{{}}.', 'Question: {q}\n{o}\nAnswer:',
             '{q}\n{o}', r'Choose the best answer. {q}' '\n{o}\n' r'Respond with \boxed{{X}} where X is the letter.',
             'The following is a multiple choice question.\n\n{q}\n\n{o}\n\nReply with only the letter.',
             '{q}\n{o}\nEnd your reply with the letter on its own line.']


def fmt(r):
    if r[0] == 'math':
        return rng.choice(MATH_STYLES).format(q=r[1].replace('{', '{{').replace('}', '}}')).replace('{{', '{').replace('}}', '}')
    if r[0] == 'mc':
        o = '\n'.join(f'{LET[i]}. {t}' for i, t in enumerate(r[2]))
        return rng.choice(MC_STYLES).replace('{q}', r[1]).replace('{o}', o).replace('{{', '{').replace('}}', '}')
    return r[1]


prompts = [fmt(r) for r in raw]
print('pool', len(prompts), flush=True)
PARENT = sys.argv[3] if len(sys.argv) > 3 else 'experiment/model'
OUT = sys.argv[4] if len(sys.argv) > 4 else 'analysis/gate_data.npz'
model, tok = H.load_model(PARENT)
lm = model.language_model.model
emb = lm.embed_tokens.weight


def hidden_last(ids):
    h = lm(mx.array([ids]))[0, -1]
    z = h @ emb.T
    mx.eval(h, z)
    z = np.array(z.astype(mx.float32))
    c = z[H.CLOSE]
    z[H.CLOSE] = -1e9
    return np.array(h.astype(mx.float32)), float(c), float(z.max())


pos, kinds = [], []
for i, p in enumerate(prompts[:N_POS]):
    ids = H.render(tok, p)
    if len(ids) > 2000:
        continue
    h, c, top = hidden_last(ids)
    pos.append((h, c, top))
    kinds.append(raw[i][0])
    if i % 100 == 0:
        print('pos', i, 'close', round(c, 2), 'top', round(top, 2), flush=True)
P = np.stack([x[0] for x in pos])
Pc = np.array([x[1] for x in pos])
Pt = np.array([x[2] for x in pos])
print('positive margin needed (top-close): mean %.2f p5 %.2f p95 %.2f max %.2f' % (
    (Pt - Pc).mean(), *np.percentile(Pt - Pc, [5, 95]), (Pt - Pc).max()), flush=True)

# Negatives: sample no-think answers on fresh prompts, then teacher-force for hidden states.
neg_items = [dict(id=f'neg{i}', benchmark='neg', prompt=p, scorer='choice', answer='A')
             for i, p in enumerate(prompts[N_POS:N_POS + N_NEG])]
rows = H.run(model, tok, neg_items, thinking=False, max_tokens=384, batch=12)
Ns, Nc, Nt = [], [], []
for it, r in zip(neg_items, rows):
    pre = H.render(tok, it['prompt'], thinking=False)
    out = tok.encode(r['text'], add_special_tokens=False)
    ids = (pre + out)[:2400]
    h = lm(mx.array([ids]))[0, len(pre) - 1:]
    z = h @ emb.T
    mx.eval(h, z)
    z = np.array(z.astype(mx.float32))
    c = z[:, H.CLOSE].copy()
    z[:, H.CLOSE] = -1e9
    Ns.append(np.array(h.astype(mx.float32)))
    Nc.append(c)
    Nt.append(z.max(1))
N = np.concatenate(Ns)
Nc = np.concatenate(Nc)
Nt = np.concatenate(Nt)
print('neg positions', len(N), 'close-top margin p50 %.2f p99 %.2f max %.2f' % (
    np.median(Nc - Nt), np.percentile(Nc - Nt, 99), (Nc - Nt).max()), flush=True)
np.savez(OUT, P=P, Pc=Pc, Pt=Pt, N=N, Nc=Nc, Nt=Nt, Ec=np.array(emb[H.CLOSE].astype(mx.float32)),
         kinds=np.array(kinds))
print('h norm pos', np.linalg.norm(P, axis=1).mean(), 'neg', np.linalg.norm(N, axis=1).mean(), 'Ec', np.linalg.norm(np.array(emb[H.CLOSE].astype(mx.float32))))
