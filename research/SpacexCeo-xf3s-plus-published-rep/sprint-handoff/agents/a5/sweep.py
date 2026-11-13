"""Offline sweep: teacher-force held-out traces through the base (new leader) and weight-edited variants; measure
next-token distribution changes at labelled positions.  Edits are applied to the weights in place (exactly what build.py does)
and restored after each config.
usage: sweep.py MODEL TEXTS RESULTS PROMPTS DIRS.pt CONFIGS.json IDS OUT.json [max_len] [n_ord]
config: {"name", "dir": key in DIRS, "alpha", "layers": [..] or "all", "mods": ["down","attn"]}
        or {"name", "upstream": alpha}  (published sparse edit, delta file upstream_delta.pt)"""
import sys, json, time, random, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import *
model, texts, results, prompts, dirsf, cfgf, ids, outf = sys.argv[1:9]
max_len = int(sys.argv[9]) if len(sys.argv) > 9 else 4096
n_ord = int(sys.argv[10]) if len(sys.argv) > 10 else 100
want = [int(x) for x in ids.split(',')]
T = {json.loads(l)['id']: json.loads(l) for l in open(texts)}
P = {p['id']: p for p in json.load(open(prompts))}
DIRS = torch.load(dirsf); CFG = json.load(open(cfgf))
tok, m = load_model(model)
L = lm(m); NL = len(L.layers)
V = tok.get_vocab()
RSET = [' check', ' Is', 'Is', ' Maybe', 'Maybe', ' But', 'But', ' re', ' look', ' try', ' However', 'However', ' Or', 'Or',
        ' assume', ' Actually', 'Actually', ' double', ' verify', ' Could', 'Could', ' Wait', 'Wait', ' wait', ' Hmm', 'Hmm',
        ' Alternatively', 'Alternatively', ' What', 'What', ' Why', ' reconsider']
RIDS = torch.tensor(sorted({V[s.replace(' ', 'Ġ')] for s in RSET if s.replace(' ', 'Ġ') in V}))
CLOSE = V['</think>']
def writers(layer_idx, mods):
    l = L.layers[layer_idx]; out = []
    if 'down' in mods: out.append(l.mlp.down_proj)
    if 'attn' in mods: out.append(l.self_attn.o_proj if hasattr(l, 'self_attn') else l.linear_attn.out_proj)
    return out
def getdir(cfg, L=None):
    key = cfg['dir'].replace('{L}', str(L)) if L is not None else cfg['dir']
    r = DIRS[key].float(); r = r / r.norm()
    for o in cfg.get('orth', []):
        o = o.replace('{L}', str(L)) if L is not None else o
        q = DIRS[o].float(); q = q / q.norm(); r = r - (r @ q) * q; r = r / r.norm()
    return r
def apply(cfg):
    saved = []
    if 'upstream' in cfg:
        D = torch.load('upstream_delta.pt')
        for li, d in D.items():
            if 'up_layers' in cfg and li not in cfg['up_layers']: continue
            w = L.layers[int(li)].mlp.down_proj.weight; saved.append((w, w.data.clone())); w.data += cfg['upstream'] * d
        return saved
    per = '{L}' in cfg['dir']
    r = None if per else getdir(cfg)
    if 'beta' in cfg:  # constant steering vector added to every embedding row (tied head => logits shift uniformly, softmax invariant)
        w = L.embed_tokens.weight; saved.append((w, w.data.clone())); w.data += cfg['beta'] * r[None, :]
        return saved
    lys = range(NL) if cfg['layers'] == 'all' else cfg['layers']
    for li in lys:
        if per: r = getdir(cfg, li)
        for mod in writers(li, cfg.get('mods', ['down', 'attn'])):
            w = mod.weight; saved.append((w, w.data.clone()))
            w.data -= cfg['alpha'] * torch.outer(r, r @ w.data)
    return saved
def restore(saved):
    for w, d in saved: w.data.copy_(d)
def logp_at(x, pos):
    with torch.no_grad():
        h = L(input_ids=x[None], use_cache=False).last_hidden_state[0, pos]
        return F.log_softmax(m.lm_head(h).float(), -1)
res = {c['name']: [] for c in CFG}; res['_base'] = []
for n, pid in enumerate(want):
    t0 = time.time()
    x, g0, lab = build(tok, P[pid]['prompt'], T[pid]['text'], max_len, with_lines=True)
    rng = random.Random(pid)
    byl = {}
    for it in lab: byl.setdefault(it[1], []).append(it)
    lab = sorted(sum([rng.sample(v, min(len(v), 120)) for k, v in byl.items()], []))
    labset = {p for p, _, _ in lab}
    cand = [p for p in range(g0 + 5, len(x) - 1) if p not in labset]
    ordp = sorted(rng.sample(cand, min(n_ord, len(cand))))
    pos = [p for p, _, _ in lab] + ordp; labels = [l for _, l, _ in lab] + ['ord'] * len(ordp)
    pt = torch.tensor(pos); nxt = x[pt + 1]
    lp0 = logp_at(x, pt)
    def stats(lp):
        p = lp.exp(); out = {}
        kl = (lp0.exp() * (lp0 - lp)).sum(-1)
        tl = lp.gather(1, nxt[:, None])[:, 0]
        prc = p[:, RIDS].sum(-1); pcl = p[:, CLOSE]
        for cls in ['ord', 'progress', 'recheck', 'close', 'groove', 'novel']:
            mk = torch.tensor([l == cls for l in labels])
            if mk.sum() == 0: continue
            out[cls] = {'n': int(mk.sum()), 'kl': kl[mk].mean().item(), 'kl_p95': kl[mk].quantile(0.95).item() if mk.sum() > 1 else kl[mk].max().item(),
                        'tok_lp': tl[mk].mean().item(), 'p_recheck': prc[mk].mean().item(), 'p_close': pcl[mk].mean().item()}
        return out
    res['_base'].append({'id': pid, **stats(lp0)})
    for c in CFG:
        sv = apply(c); lp = logp_at(x, pt); restore(sv)
        res[c['name']].append({'id': pid, **stats(lp)})
    print(n, pid, len(x), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(outf, 'w'))
# aggregate (mean over traces weighted by n)
agg = {}
for k, rows in res.items():
    a = {}
    for cls in ['ord', 'progress', 'recheck', 'close', 'groove', 'novel']:
        rr = [r[cls] for r in rows if cls in r]; N = sum(r['n'] for r in rr)
        if N: a[cls] = {f: sum(r[f] * r['n'] for r in rr) / N for f in ['kl', 'kl_p95', 'tok_lp', 'p_recheck', 'p_close']}
    agg[k] = a
json.dump({'per_trace': res, 'agg': agg}, open(outf, 'w'), indent=0)
for k, a in agg.items(): print(k, json.dumps({c: {f: round(v, 4) for f, v in d.items()} for c, d in a.items()}))
