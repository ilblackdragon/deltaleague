"""Loop/copy analysis and g-sweep summaries from tf.py output. usage: analyze2.py RUN.npz"""
import sys, numpy as np, collections
d = np.load(sys.argv[1], allow_pickle=True); C = [str(c) for c in d['confs']]; ci = {c: i for i, c in enumerate(C)}
pids = sorted({k.split('__')[0] for k in d.files if '__' in k})
D = {int(p): {k.split('__')[1]: d[k] for k in d.files if k.startswith(p + '__')} for p in pids}
def outcome(m): return 'correct' if m[1] == 1 else ('trunc' if m[2] else 'wrong')
G = collections.defaultdict(list)
for p, x in D.items(): G[outcome(x['meta'])].append(p)
print({k: len(v) for k, v in G.items()}, 'inexact rows', np.mean(np.concatenate([x['inexact'] for x in D.values()])))
N = 12
def copymask(tk):
    """position t is 'copy' if the N-gram ending at t-1 occurred before and the actual token equals its earlier continuation"""
    seen = {}; m = np.zeros(len(tk), bool); inrep = np.zeros(len(tk), bool)
    for t in range(N, len(tk)):
        key = tuple(tk[t - N:t])
        if key in seen:
            inrep[t] = True; m[t] = tk[seen[key]] == tk[t]
        seen[key] = t
    return m, inrep
GS = [c for c in C if c.startswith('g') and len(c) == 5]
print('\n== loop escape: share of positions that are verbatim copies (12-gram seen before, same continuation), and per-g mean escape prob 1-q(copy token)')
for g, ps in G.items():
    ms = [copymask(D[p]['tokens']) for p in ps]
    share = np.mean([m.mean() for m, _ in ms]); late = np.mean([m[len(m) // 2:].mean() for m, _ in ms])
    row = []
    for c in GS:
        q = np.concatenate([np.exp(D[p]['lq'][ci[c]][m]) for p, (m, _) in zip(ps, ms)])
        row.append(f'{c}:{1 - q.mean():.4f}')
    print(f'{g:8s} copy-share={share:.3f} (2nd half {late:.3f})  escape/token ' + ' '.join(row))
    # expected run length of a copy streak if escape prob e per token: 1/e
print('\n== per-g on-policy-trace summaries (mean over traces): entropy, 1-p(top) "branch" rate, hazards')
for g, ps in G.items():
    for c in GS + [c for c in C if 'think' in c or 'maybe' in c]:
        x = [D[p] for p in ps]
        H = np.mean([y['H'][ci[c]].mean() for y in x]); st = np.mean([y['pthink'][ci[c]].sum() for y in x])
        sh = np.mean([y['phedge'][ci[c]].sum() for y in x]); z = np.mean([1000 * (y['lq'][ci[c]] < -60).mean() for y in x])
        n1 = np.mean([(y['n'][ci[c]] == 1).mean() for y in x])
        # close hazard excluding the true close position and the 30 tokens before it (premature-close hazard)
        pre = []
        for y in x:
            w = np.where(y['tokens'] == 248069)[0]; e = (w[0] - 30) if len(w) else len(y['tokens'])
            pre.append(y['pthink'][ci[c]][:max(e, 0)].sum())
        print(f'{g:8s} {c:16s} H={H:.3f} P(n=1)={n1:.3f} zero/1k={z:6.2f} sum_p_think={st:7.4f} premature={np.mean(pre):7.4f} sum_p_hedge={sh:6.2f}')
