"""Teacher-force traces through a model on CPU; save residual stream (output of every decoder layer) at labelled positions.
usage: collect.py MODEL TEXTS RESULTS PROMPTS OUT.pt IDS(comma) [max_len] [n_ord]"""
import sys, json, time, random, torch
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
model, texts, results, prompts, out, ids = sys.argv[1:7]
max_len = int(sys.argv[7]) if len(sys.argv) > 7 else 6144
n_ord = int(sys.argv[8]) if len(sys.argv) > 8 else 60
want = [int(x) for x in ids.split(',')]
T = {json.loads(l)['id']: json.loads(l) for l in open(texts)}
R = {r['id']: r for r in json.load(open(results))['rows']}
P = {p['id']: p for p in json.load(open(prompts))}
tok, m = load_model(model)
L = lm(m); layers = L.layers
cache = {}
def mk(i):
    def h(mod, inp, outp):
        o = outp[0] if isinstance(outp, tuple) else outp
        cache[i] = o[0, cache['pos']].detach().to(torch.float16)
    return h
for i, l in enumerate(layers): l.register_forward_hook(mk(i))
recs = []
for n, pid in enumerate(want):
    t0 = time.time()
    x, g0, lab = build(tok, P[pid]['prompt'], T[pid]['text'], max_len, with_lines=True)
    rng = random.Random(pid)
    CAP = {'novel': 80, 'progress': 80, 'recheck': 80, 'groove': 80}
    byl = {}
    for it in lab: byl.setdefault(it[1], []).append(it)
    lab = sorted(sum([rng.sample(v, min(len(v), CAP.get(k, 10**6))) for k, v in byl.items()], []))
    labset = {p for p, _, _ in lab}
    cand = [p for p in range(g0 + 5, len(x) - 1) if p not in labset and all(abs(p - q) > 2 for q in labset)]
    ordp = sorted(rng.sample(cand, min(n_ord, len(cand))))
    allpos = [p for p, _, _ in lab] + ordp
    labels = [l for _, l, _ in lab] + ['ord'] * len(ordp)
    words = [w for _, _, w in lab] + [''] * len(ordp)
    cache['pos'] = torch.tensor(allpos)
    with torch.no_grad():
        L(input_ids=x[None], use_cache=False)
    H = torch.stack([cache[i] for i in range(len(layers))], 1)  # [npos, nlayer, d]
    recs.append({'id': pid, 'trunc': R[pid]['truncated'], 'score': R[pid]['score'], 'pos': allpos, 'g0': g0, 'len': len(x),
                 'labels': labels, 'words': words, 'next': [int(x[p + 1]) for p in allpos], 'H': H})
    print(n, pid, len(x), 'labs', {k: labels.count(k) for k in set(labels)}, f'{time.time()-t0:.1f}s', flush=True)
    if n % 8 == 7: torch.save(recs, out)
torch.save(recs, out)
print('saved', out)
