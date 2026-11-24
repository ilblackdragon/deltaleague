"""Gap analysis: gap = (best other penalised logit) - (</think> logit), at newline positions; nucleus threshold effects.
usage: gap.py NPZ_DIR RESULT.json"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load, p_close, value_curve
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model'))
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
D = load(sys.argv[1], sys.argv[2])
def q(x): return ' '.join(f'{v:6.2f}' for v in np.percentile(x, [5, 25, 50, 75, 95])) if len(x) else 'n/a'
G = {'closed: natural close pos': [], 'closed: newlines before close': [], 'never: newlines, answer not known': [],
     'never: newlines, answer known (probe ok)': [], 'all: non-newline positions': []}
LC = {k: [] for k in G}
for d in D:
    T = len(d['lc']); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
    gap = d['top'][:, 0] - d['lc']; V = value_curve(d)
    if d['closed']:
        G['closed: natural close pos'].append(gap[-1:]); LC['closed: natural close pos'].append(d['lc'][-1:])
        m = nl.copy(); m[-1] = False
        G['closed: newlines before close'].append(gap[m]); LC['closed: newlines before close'].append(d['lc'][m])
    else:
        m = nl & (V == 0); G['never: newlines, answer not known'].append(gap[m]); LC['never: newlines, answer not known'].append(d['lc'][m])
        m = nl & (V > 0); G['never: newlines, answer known (probe ok)'].append(gap[m]); LC['never: newlines, answer known (probe ok)'].append(d['lc'][m])
    G['all: non-newline positions'].append(gap[~nl]); LC['all: non-newline positions'].append(d['lc'][~nl])
print('percentiles 5/25/50/75/95')
for k in G:
    g = np.concatenate(G[k]) if G[k] else np.array([]); l = np.concatenate(LC[k]) if LC[k] else np.array([])
    print(f'{k:42s} n={len(g):6d} gap: {q(g)} | lc: {q(l)} | frac gap<1: {np.mean(g < 1) if len(g) else 0:.3f} gap<2: {np.mean(g < 2) if len(g) else 0:.3f}')
# threshold: smallest additive boost b such that p_close>0 at a newline (nucleus entry)
for k in ['never: newlines, answer not known', 'never: newlines, answer known (probe ok)', 'closed: newlines before close']:
    pass
# per-trace: minimum gap over newline positions (how far each never-closed trace is from ever closing)
for lab, sel in [('never-closed', [d for d in D if not d['closed']]), ('closed (before close)', [d for d in D if d['closed']])]:
    mg = []
    for d in sel:
        T = len(d['lc']); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
        if d['closed']: nl[-1] = False
        g = (d['top'][:, 0] - d['lc'])[nl]
        mg.append(np.sort(g)[:5].mean() if len(g) >= 5 else (g.min() if len(g) else np.nan))
    print(f'{lab}: per-trace mean of 5 smallest newline gaps: {q(np.array(mg)[~np.isnan(mg)])}')
# additive boost needed for sum p >= 0.5 per never-closed trace
for b in [0.5, 1, 1.5, 2, 3]:
    s = [p_close(d['lc'] + b, d['top']).sum() for d in D if not d['closed']]
    s2 = [p_close(d['lc'][:-1] + b, d['top'][:-1]).sum() for d in D if d['closed']]
    print(f'const +{b}: never-closed: median sum p {np.median(s):.3f}, frac traces sum p>0.7 {np.mean(np.array(s) > .7):.2f} | closed (pre-close): median sum p {np.median(s2):.3f}, frac>0.7 {np.mean(np.array(s2) > .7):.2f}')
