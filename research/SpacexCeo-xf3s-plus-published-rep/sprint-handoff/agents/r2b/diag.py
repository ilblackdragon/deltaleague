import json, re, sys, os, numpy as np
sys.path.insert(0, '/home/user/delta/agents/a1')
from analyze import enrich
R = '/home/user/delta/agents/a1/res/'; L = '/home/user/delta/results/'; A5 = '/home/user/delta/agents/a5/results/'
SETS = {'ref': [R+'ref_xf3s_x4.json', R+'c_xf3s.json', R+'a1_xf3s_s2.json'],
        'h05': [L+'h_rep05.json', L+'h_rep05_s2.json'],
        'x10': [A5+'a5_xrep10.json', A5+'a5_xrep10_s2.json']}
BOX = re.compile(r'\\boxed\{')
def mcletter(ans):
    m = re.findall(r'\\boxed\{\s*(?:\\text\{)?([A-J])', ans) or re.findall(r'(?:answer is|Answer:)\s*\(?([A-J])\b', ans) or re.findall(r'(?:^|\n)\s*\(?([A-J])[.)]?\s*$', ans)
    return m[-1] if m else None
def load(paths):
    out = []
    for p in paths:
        d = enrich(p); T = {}
        for line in open(p[:-5] + '.texts.jsonl'):
            t = json.loads(line); T[(t['id'], t['sample'])] = t['text']
        for r in d['rows']:
            txt = T[(r['id'], r['sample'])]; r['pid'] = r['id'] % 10000; r['src'] = os.path.basename(p)
            r['healthy'] = bool(r['closed'] and not r['truncated'] and r['think_tok'] >= 50)
            if '</think>' in txt:
                th, ans = txt.split('</think>', 1)[0], txt.split('</think>')[-1]
                r['boxed'] = bool(BOX.search(ans)); r['letter'] = mcletter(ans)
                r['think_boxed'] = bool(BOX.search(th)); r['ans_text'] = ans; r['think_tail'] = th[-1500:]
                r['ans_steps'] = len(re.findall(r'(?m)^\s*(?:\*\*|#+|\d+\.)', ans))
            out.append(r)
    return out
D = {k: load(v) for k, v in SETS.items()}
pickle_path = '/home/user/delta/agents/r2b/diag_rows.json'
json.dump({k: [{kk: vv for kk, vv in r.items() if kk not in ('answer_tail',)} for r in v] for k, v in D.items()}, open(pickle_path, 'w'))
def byp(rows):
    b = {}
    for r in rows: b.setdefault(r['pid'], []).append(r)
    return b
P = {k: byp(v) for k, v in D.items()}
ids = sorted(P['ref'])
def hshare(rs): return np.mean([r['healthy'] for r in rs])
def strict(rs): return np.mean([r['healthy'] and r['score'] > 0 for r in rs])
bench = {i: P['ref'][i][0]['bench'] for i in ids}
print('== class by ref healthy share; per class: healthy share, acc|healthy (gen-pooled), strict ==')
for lo, hi, nm in [(0.75, 1.01, 'normally closed (ref h>=.75)'), (0.25, 0.75, 'mixed'), (-0.01, 0.25, 'mostly unclosed (ref h<.25)')]:
    cls = [i for i in ids if lo <= hshare(P['ref'][i]) < hi]
    line = f'{nm:32s} n={len(cls):3d}'
    for k in ('ref', 'h05', 'x10'):
        rs = [r for i in cls for r in P[k][i]]; h = [r for r in rs if r['healthy']]
        # prompt-averaged acc|healthy
        pa = [np.mean([r['score'] for r in P[k][i] if r['healthy']]) for i in cls if any(r['healthy'] for r in P[k][i])]
        line += f' | {k}: h {100*np.mean([r["healthy"] for r in rs]):5.1f} acc|h {100*np.mean([r["score"] for r in h]):5.1f} (pavg {100*np.mean(pa):5.1f}) strict {100*np.mean([strict(P[k][i]) for i in cls]):5.1f}'
    print(line)
print('\n== normally closed prompts by bench ==')
cls = [i for i in ids if hshare(P['ref'][i]) >= 0.75]
for bb in ('math500', 'gsm8k', 'mmlu_pro', 'arc_c'):
    c2 = [i for i in cls if bench[i] == bb]; line = f'{bb:9s} n={len(c2):3d}'
    for k in ('ref', 'h05', 'x10'):
        rs = [r for i in c2 for r in P[k][i]]; h = [r for r in rs if r['healthy']]
        line += f' | {k}: h {100*np.mean([r["healthy"] for r in rs]):5.1f} acc|h {100*np.mean([r["score"] for r in h]):5.1f} strict {100*np.mean([strict(P[k][i]) for i in c2]):5.1f}'
    print(line)
print('\n== all prompts by bench (strict / acc|h / healthy) ==')
for bb in ('math500', 'gsm8k', 'mmlu_pro', 'arc_c'):
    c2 = [i for i in ids if bench[i] == bb]; line = f'{bb:9s} n={len(c2):3d}'
    for k in ('ref', 'h05', 'x10'):
        rs = [r for i in c2 for r in P[k][i]]; h = [r for r in rs if r['healthy']]
        line += f' | {k}: strict {100*np.mean([strict(P[k][i]) for i in c2]):5.1f} acc|h {100*np.mean([r["score"] for r in h]):5.1f} h {100*np.mean([r["healthy"] for r in rs]):5.1f}'
    print(line)
print('\n== healthy-gen features: correct vs wrong (ref | h05 | x10) ==')
for k in ('ref', 'h05', 'x10'):
    h = [r for r in D[k] if r['healthy']]
    for nm, f in [('think_tok med', lambda r: r['think_tok']), ('ans_tok med', lambda r: r['ans_tok'])]:
        c = [f(r) for r in h if r['score'] > 0]; w = [f(r) for r in h if r['score'] == 0]
        print(f'{k} {nm}: correct {np.median(c):.0f}  wrong {np.median(w):.0f}')
    for bb in ('math500', 'gsm8k', 'mmlu_pro', 'arc_c'):
        hb = [r for r in h if r['bench'] == bb]; w = [r for r in hb if r['score'] == 0]
        if bb in ('mmlu_pro', 'arc_c'):
            print(f'  {k} {bb}: healthy {len(hb)} wrong {len(w)}; wrong w/o parsable letter {sum(r["letter"] is None for r in w)} ({100*np.mean([r["letter"] is None for r in hb]):.1f}% of healthy); wrong with letter {sum(r["letter"] is not None for r in w)}')
        else:
            print(f'  {k} {bb}: healthy {len(hb)} wrong {len(w)}; wrong w/o \\boxed {sum(not r["boxed"] for r in w)} ({100*np.mean([not r["boxed"] for r in hb]):.1f}% of healthy)')
