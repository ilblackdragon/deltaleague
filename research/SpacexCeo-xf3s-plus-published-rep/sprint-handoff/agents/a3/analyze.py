"""Summaries of tf.py output. usage: analyze.py RUN.npz [more.npz]"""
import sys, numpy as np, collections, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
voc = tok.get_vocab(); inv = {i: s for s, i in voc.items()}
D = {}
for f in sys.argv[1:]:
    d = np.load(f, allow_pickle=True); C = [str(c) for c in d['confs']]
    pids = sorted({k.split('__')[0] for k in d.files if '__' in k})
    for p in pids: D[int(p)] = {k.split('__')[1]: d[k] for k in d.files if k.startswith(p + '__')}
ci = {c: i for i, c in enumerate(C)}; ON = ci['g1.40']
def outcome(m): return 'correct' if m[1] == 1 else ('trunc' if m[2] else 'wrong')
groups = collections.defaultdict(list)
for p, x in D.items(): groups[outcome(x['meta'])].append(p)
print({k: len(v) for k, v in groups.items()})
DIG = {i for s, i in voc.items() if s.lstrip('Ġ').isdigit()}
out = {}
# A. on-policy position stats
print('\n== A. on-policy (g=1.4) stats by outcome and position bin: H, nucleus n, P(n=1), ppen, pen_act, top20pen')
bins = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8192)]
for g, ps in groups.items():
    for a, b in bins:
        sel = [(x, slice(a, min(b, len(x['rank'])))) for x in (D[p] for p in ps) if len(x['rank']) > a]
        if not sel: continue
        cat = lambda k, c=None: np.concatenate([x[k][c][s] if c is not None else x[k][s] for x, s in sel])
        print(f'{g:8s} {a:5d}-{b:5d} H={cat("H", ON).mean():.3f} n={cat("n", ON).mean():.2f} P(n=1)={(cat("n", ON) == 1).mean():.2f} '
              f'ppen={cat("ppen", ON).mean():.3f} pen_act={cat("pen_act").mean():.3f} top20pen={cat("top20pen").mean():.3f} N={len(cat("rank"))}')
# B/C. per config: entropy, n, zero-prob tokens, per-token log-ratio vs on-policy, hazards
print('\n== B/C/D/E. per config (means over traces of each outcome)')
print('config | outcome | H | n | P(n=1) | zero-prob tok/1k | dLL/tok (finite) | sum p(</think>) | 1-exp(-sum) | sum p(hedge)')
for c in C:
    for g, ps in groups.items():
        H = np.mean([D[p]['H'][ci[c]].mean() for p in ps]); n = np.mean([D[p]['n'][ci[c]].mean() for p in ps])
        n1 = np.mean([(D[p]['n'][ci[c]] == 1).mean() for p in ps])
        z = np.mean([1000 * (D[p]['lq'][ci[c]] < -60).mean() for p in ps])
        dl = np.mean([np.where((D[p]['lq'][ci[c]] > -60) & (D[p]['lq'][ON] > -60), D[p]['lq'][ci[c]] - D[p]['lq'][ON], 0).mean() for p in ps])
        st = np.mean([D[p]['pthink'][ci[c]].sum() for p in ps]); sh = np.mean([D[p]['phedge'][ci[c]].sum() for p in ps])
        pc = np.mean([1 - np.exp(-D[p]['pthink'][ci[c]].sum()) for p in ps])
        out[(c, g)] = dict(H=H, n=n, z=z, dl=dl, st=st, sh=sh)
        print(f'{c:16s} {g:8s} {H:.3f} {n:.2f} {n1:.2f} {z:6.2f} {dl:+.4f} {st:8.4f} {pc:.3f} {sh:7.2f}')
# D2. at the actual </think> position of closed traces
print('\n== D2. q(</think>) at the position where the trace actually closed (closed traces)')
for c in C:
    v = []
    for p, x in D.items():
        w = np.where(x['tokens'] == 248069)[0]
        if len(w): v.append(np.exp(x['lq'][ci[c]][w[0]]))
    print(f'{c:16s} mean {np.mean(v):.3f} median {np.median(v):.3f} zero {np.mean(np.array(v) < 1e-20):.2f} n={len(v)}')
# z_think vs z_top at close positions and elsewhere
gap = []; gap_else = []
for p, x in D.items():
    w = np.where(x['tokens'] == 248069)[0]
    zz = x['zthink'] - x['ztop']
    if len(w): gap.append(zz[w[0]])
    gap_else.append(zz[:len(zz) if not len(w) else w[0]])
ge = np.concatenate(gap_else)
print('gain-1 gap z(</think>)-z(top) at close: ', np.round(np.percentile(gap, [10, 50, 90]), 2),
      ' elsewhere pct50/99/99.9:', np.round(np.percentile(ge, [50, 99, 99.9]), 2), ' frac>-1:', (ge > -1).mean(), 'frac>-2:', (ge > -2).mean())
# F. penalty damage: positions where the actual token was penalized
print('\n== F. presence-penalty damage on the actual token: log q(g1.4) - log q(g1.4, no penalty)')
for g, ps in groups.items():
    for cls in ['all', 'digit', 'nondigit']:
        dd = []; frac = []
        for p in ps:
            x = D[p]; m = x['pen_act'] & (x['lq'][ON] > -60) & (x['lq'][ci['g1.40_nopen']] > -60)
            isd = np.array([t in DIG for t in x['tokens']])
            if cls == 'digit': m &= isd
            if cls == 'nondigit': m &= ~isd
            dd.append((x['lq'][ON] - x['lq'][ci['g1.40_nopen']])[m]); frac.append(m.mean())
        dd = np.concatenate(dd)
        print(f'{g:8s} {cls:8s} share-of-positions={np.mean(frac):.3f} mean dlogq={dd.mean():+.4f} P(dlogq<-0.1)={(dd < -0.1).mean():.3f} P(dlogq<-1)={(dd < -1).mean():.4f}')
    nov = np.concatenate([(D[p]['lq'][ON] - D[p]['lq'][ci['g1.40_nopen']])[~D[p]['pen_act'] & (D[p]['lq'][ON] > -60) & (D[p]['lq'][ci['g1.40_nopen']] > -60)] for p in ps])
    print(f'{g:8s} novel    mean dlogq={nov.mean():+.4f}  (actual token not yet used)')
