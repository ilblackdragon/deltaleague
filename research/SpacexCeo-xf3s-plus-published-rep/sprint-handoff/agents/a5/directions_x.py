"""Groove (verbatim/digit-normalised line repetition) directions on b8af1dbc traces.
gn: groove line starts - novel line starts, truncated traces only, both restricted to positions >= MINPOS (controls for depth)
gc: groove line starts (trunc) - novel line starts in CLOSED traces (any position)
usage: directions_x.py ACTS.pt DIRS_OUT.pt OUT.json"""
import sys, json, torch, random
acts, outd, outj = sys.argv[1:4]
recs = torch.load(acts); NL = recs[0]['H'].shape[1]; MINPOS = 1500
def gather(sel):
    xs, ti = [], []
    for k, r in enumerate(recs):
        m = [i for i, (l, p) in enumerate(zip(r['labels'], r['pos'])) if sel(r, l, p - r['g0'])]
        if m: xs.append(r['H'][m].float()); ti += [k] * len(m)
    return (torch.cat(xs), torch.tensor(ti)) if xs else (torch.zeros(0, NL, 1024), torch.zeros(0, dtype=torch.long))
C = {'groove': gather(lambda r, l, p: l == 'groove' and r['trunc'] and p >= MINPOS),
     'novel_t': gather(lambda r, l, p: l == 'novel' and r['trunc'] and p >= MINPOS),
     'novel_c': gather(lambda r, l, p: l == 'novel' and not r['trunc']),
     'recheck': gather(lambda r, l, p: l == 'recheck'), 'progress': gather(lambda r, l, p: l == 'progress'),
     'ord_t': gather(lambda r, l, p: l == 'ord' and r['trunc'] and p >= MINPOS), 'ord_c': gather(lambda r, l, p: l == 'ord' and not r['trunc'])}
print({k: int(v[0].shape[0]) for k, v in C.items()})
idx = list(range(len(recs))); random.Random(0).shuffle(idx); A = set(idx[::2]); B = set(idx[1::2])
def half(name, S):
    X, t = C[name]; m = torch.tensor([int(i) in S for i in t], dtype=torch.bool); return X[m]
def auc(s1, s0):
    s = torch.cat([s1, s0]); r = s.argsort().argsort().float() + 1
    return ((r[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))).item()
PAIRS = {'gn': ('groove', 'novel_t'), 'gc': ('groove', 'novel_c'), 'rpx': ('recheck', 'progress'), 'stx': ('ord_t', 'ord_c')}
d13 = torch.tensor(json.load(open('/home/clore/dl/xf3s/doubt_direction_L13.json'))['values']); d13 /= d13.norm()
up = torch.load('upstream_rank1.pt'); old = torch.load('res/dirs.pt')
D = {}; out = {'n': {k: int(v[0].shape[0]) for k, v in C.items()}, 'layers': {}}
for L in range(NL):
    o = {}
    for key, (c1, c0) in PAIRS.items():
        X1, X0 = C[c1][0][:, L], C[c0][0][:, L]
        if len(X1) < 6 or len(X0) < 6: continue
        d = X1.mean(0) - X0.mean(0); D[f'{key}_L{L}'] = d / d.norm()
        a1, a0, b1, b0 = half(c1, A)[:, L], half(c0, A)[:, L], half(c1, B)[:, L], half(c0, B)[:, L]
        ok = min(len(a1), len(a0), len(b1), len(b0)) >= 3
        dA = a1.mean(0) - a0.mean(0) if ok else d; dB = b1.mean(0) - b0.mean(0) if ok else d
        u = dA / dA.norm(); un = d / d.norm()
        o[key] = {'norm': d.norm().item(), 'rel': d.norm().item() / torch.cat([X1, X0]).norm(dim=-1).mean().item(),
                  'split_cos': torch.cosine_similarity(dA, dB, 0).item() if ok else float('nan'),
                  'auc_heldout': auc(b1 @ u, b0 @ u) if ok else float('nan'),
                  'mu1': (X1 @ un).mean().item(), 'mu0': (X0 @ un).mean().item(), 'sd0': (X0 @ un).std().item(),
                  'cos_d13': (un @ d13).item(), 'cos_up15': (un @ up['15'][0]).item(), 'cos_up19': (un @ up['19'][0]).item(),
                  'cos_old_rp': (un @ old[f'rp_L{L}']).item(), 'cos_old_stuck': (un @ old[f'stuck_p_L{L}']).item()}
    out['layers'][L] = o
    print(L, {k: {f: round(v, 2) for f, v in vv.items()} for k, vv in o.items()}, flush=True)
D.update({k: v for k, v in old.items() if k.startswith(('rp_', 'cp_', 'up'))}); D['d13'] = d13
torch.save(D, outd); json.dump(out, open(outj, 'w'), indent=0)
