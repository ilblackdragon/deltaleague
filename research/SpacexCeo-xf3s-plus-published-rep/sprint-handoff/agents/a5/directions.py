"""Difference-of-means directions per layer from collected activations; split-half stability; linear separability
(held-out trace halves); geometry vs the published rank-1 edit.  usage: directions.py ACTS.pt OUT_DIRS.pt OUT.json"""
import sys, json, torch, random
acts, outd, outj = sys.argv[1:4]
recs = torch.load(acts)
NL = recs[0]['H'].shape[1]
def gather(sel):
    """sel(rec, label) -> bool ; returns X [n, NL, d] float32 and trace index"""
    xs, ti = [], []
    for k, r in enumerate(recs):
        m = [i for i, l in enumerate(r['labels']) if sel(r, l)]
        if m: xs.append(r['H'][m].float()); ti += [k] * len(m)
    return torch.cat(xs), torch.tensor(ti)
C = {'recheck': gather(lambda r, l: l == 'recheck'), 'progress': gather(lambda r, l: l == 'progress'),
     'close': gather(lambda r, l: l == 'close'), 'ord': gather(lambda r, l: l == 'ord'),
     'prog_trunc': gather(lambda r, l: l == 'progress' and r['trunc']), 'prog_closed': gather(lambda r, l: l == 'progress' and not r['trunc']),
     'ord_trunc': gather(lambda r, l: l == 'ord' and r['trunc']), 'ord_closed': gather(lambda r, l: l == 'ord' and not r['trunc'])}
print({k: v[0].shape[0] for k, v in C.items()})
ntr = len(recs); idx = list(range(ntr)); random.Random(0).shuffle(idx)
A = set(idx[:ntr // 2]); B = set(idx[ntr // 2:])
def half(name, S):
    X, t = C[name]; m = torch.tensor([int(i) in S for i in t]); return X[m]
PAIRS = {'rp': ('recheck', 'progress'), 'ro': ('recheck', 'ord'), 'cp': ('close', 'progress'), 'co': ('close', 'ord'),
         'stuck_p': ('prog_trunc', 'prog_closed'), 'stuck_o': ('ord_trunc', 'ord_closed')}
up = torch.load('upstream_rank1.pt')
out = {'n': {k: int(v[0].shape[0]) for k, v in C.items()}, 'layers': {}}
D = {}
def auc(s1, s0):
    s = torch.cat([s1, s0]); r = s.argsort().argsort().float() + 1
    return ((r[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))).item()
def logreg(Xa, ya, Xb, yb, steps=300, wd=1e-3):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-4
    Xa = (Xa - mu) / sd; Xb = (Xb - mu) / sd
    w = torch.zeros(Xa.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)
    pw = torch.tensor([(ya == 0).sum() / (ya == 1).sum()])
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w + b, ya, pos_weight=pw) + wd * (w ** 2).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): s = Xb @ w + b
    return auc(s[yb == 1], s[yb == 0])
for L in range(NL):
    o = {}
    for key, (c1, c0) in PAIRS.items():
        X1, X0 = C[c1][0][:, L], C[c0][0][:, L]
        if len(X1) < 4 or len(X0) < 4: continue
        d = X1.mean(0) - X0.mean(0); D[f'{key}_L{L}'] = d / d.norm()
        dA = half(c1, A)[:, L].mean(0) - half(c0, A)[:, L].mean(0); dB = half(c1, B)[:, L].mean(0) - half(c0, B)[:, L].mean(0)
        resn = torch.cat([X1, X0]).norm(dim=-1).mean().item()
        u = dA / dA.norm()
        o[key] = {'norm': d.norm().item(), 'rel': d.norm().item() / resn, 'split_cos': torch.cosine_similarity(dA, dB, 0).item(),
                  'auc_1d_heldout': auc(half(c1, B)[:, L] @ u, half(c0, B)[:, L] @ u),
                  'proj_mean_c1': (X1 @ (d / d.norm())).mean().item(), 'proj_mean_c0': (X0 @ (d / d.norm())).mean().item(),
                  'proj_sd_c0': (X0 @ (d / d.norm())).std().item()}
        if key in ('rp', 'cp', 'stuck_p'):
            Xa = torch.cat([half(c1, A)[:, L], half(c0, A)[:, L]]); ya = torch.cat([torch.ones(len(half(c1, A))), torch.zeros(len(half(c0, A)))])
            Xb = torch.cat([half(c1, B)[:, L], half(c0, B)[:, L]]); yb = torch.cat([torch.ones(len(half(c1, B))), torch.zeros(len(half(c0, B)))])
            o[key]['auc_logreg_heldout'] = logreg(Xa, ya, Xb, yb)
        for k2 in ('15', '19'): o[key][f'cos_up{k2}'] = torch.cosine_similarity(d, up[k2][0], 0).item()
    out['layers'][L] = o
    print(L, {k: {f: round(v, 3) for f, v in vv.items()} for k, vv in o.items()}, flush=True)
# cross-layer cosine of rp directions, and rp vs cp / stuck
out['cos_rp_cp'] = [torch.dot(D[f'rp_L{L}'], D[f'cp_L{L}']).item() for L in range(NL) if f'cp_L{L}' in D]
out['cos_rp_stuck'] = [torch.dot(D[f'rp_L{L}'], D[f'stuck_p_L{L}']).item() for L in range(NL)]
out['cos_rp_adjacent'] = [torch.dot(D[f'rp_L{L}'], D[f'rp_L{L+1}']).item() for L in range(NL - 1)]
for k in ('cos_rp_cp', 'cos_rp_stuck', 'cos_rp_adjacent'): print(k, [round(x, 2) for x in out[k]])
D['up15'] = up['15'][0]; D['up19'] = up['19'][0]
torch.save(D, outd); json.dump(out, open(outj, 'w'), indent=0)
