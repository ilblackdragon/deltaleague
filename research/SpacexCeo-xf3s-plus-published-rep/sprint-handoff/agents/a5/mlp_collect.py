"""Collect MLP activations (down_proj inputs, 3584-d) at groove / novel line starts for layers 12..22 (local, x_model).
usage: mlp_collect.py IDS OUT.pt"""
import sys, json, random, torch
sys.path.insert(0, 'box'); from common import *
ids = [int(x) for x in sys.argv[1].split(',')]; out = sys.argv[2]
tok, m = load_model('x_model', threads=3); L = lm(m)
T = {json.loads(l)['id']: json.loads(l) for l in open('results/c_xf3s.texts.jsonl')}
R = {r['id']: r for r in json.load(open('results/c_xf3s.json'))['rows']}
P = {p['id']: p for p in json.load(open('/home/user/delta/agents/a1/proxy400.json'))}
LAY = list(range(12, 23)); cache = {}
def mk(i):
    def h(mod, inp): cache[i] = inp[0][0, cache['pos']].detach().half()
    return h
for i in LAY: L.layers[i].mlp.down_proj.register_forward_pre_hook(mk(i))
recs = []
for pid in ids:
    x, g0, lab = build(tok, P[pid]['prompt'], T[pid]['text'], 6144, with_lines=True)
    rng = random.Random(pid); lab = [it for it in lab if it[1] in ('groove', 'novel', 'recheck', 'progress')]
    byl = {}
    for it in lab: byl.setdefault(it[1], []).append(it)
    lab = sorted(sum([rng.sample(v, min(len(v), 150)) for v in byl.values()], []))
    cache['pos'] = torch.tensor([p for p, _, _ in lab])
    with torch.no_grad(): L(input_ids=x[None], use_cache=False)
    recs.append({'id': pid, 'trunc': R[pid]['truncated'], 'labels': [l for _, l, _ in lab], 'pos': [p - g0 for p, _, _ in lab],
                 'A': {i: cache[i] for i in LAY}})
    print(pid, len(x), {k: len(v) for k, v in byl.items()}, flush=True)
    torch.save(recs, out)
