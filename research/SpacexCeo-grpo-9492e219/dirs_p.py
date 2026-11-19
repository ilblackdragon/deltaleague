"""Directions from collect_p.py activations. For each pair (target, contrast) and layer: difference of means, position-matched
(1k-token bins of absolute generation position) difference of means, held-out AUC (prompt-id halves), projection means per label,
logit lens of the direction through the tied embedding (top +/- tokens).  usage: dirs_p.py ACTS.pt MODEL_DIR OUT_DIRS.pt"""
import sys, json, torch, collections
from safetensors import safe_open
from transformers import AutoTokenizer
acts, mdir, outd = sys.argv[1:4]
A = torch.load(acts); LAY = A['layers']; recs = A['recs']
X = collections.defaultdict(list); POS = collections.defaultdict(list); PID = collections.defaultdict(list)
for r in recs:
    for j, l in enumerate(r['labels']):
        X[l].append(r['H'][j]); POS[l].append(r['pos'][j]); PID[l].append(r['id'])
X = {l: torch.stack(v).float() for l, v in X.items()}; POS = {l: torch.tensor(v) for l, v in POS.items()}; PID = {l: torch.tensor(v) for l, v in PID.items()}
print({l: tuple(v.shape) for l, v in X.items()})
for l in X: print(l, 'pos median', int(POS[l].median()), 'p10', int(POS[l].float().quantile(.1)), 'p90', int(POS[l].float().quantile(.9)))
f = safe_open(f'{mdir}/model.safetensors', 'pt'); E = f.get_tensor('model.language_model.embed_tokens.weight').float()
nw = f.get_tensor('model.language_model.norm.weight').float(); tok = AutoTokenizer.from_pretrained(mdir)
def auc(s1, s0):
    s = torch.cat([s1, s0]); r = s.argsort().argsort().float() + 1
    return ((r[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))).item()
def matched(t, c, li):
    bt, bc = POS[t] // 1000, POS[c] // 1000; ds, ws = [], []
    for b in set(bt.tolist()) & set(bc.tolist()):
        mt, mc = bt == b, bc == b
        if mt.sum() >= 20 and mc.sum() >= 20: ds.append(X[t][mt, li].mean(0) - X[c][mc, li].mean(0)); ws.append(min(mt.sum(), mc.sum()).item())
    if not ds: return None, 0
    w = torch.tensor(ws, dtype=torch.float); return (torch.stack(ds) * w[:, None]).sum(0) / w.sum(), int(w.sum())
PAIRS = {'enum': ('enum_tail', 'comp_ok'), 'over': ('ans_over', 'ans_ok')}
D = {}
for name, (t, c) in PAIRS.items():
    print(f'\n=== {name}: {t} (n {len(X[t])}) vs {c} (n {len(X[c])})')
    for li, L in enumerate(LAY):
        u = X[t][:, li].mean(0) - X[c][:, li].mean(0); un = u / u.norm()
        um, nm = matched(t, c, li); cosm = (um / um.norm() @ un).item() if um is not None else float('nan')
        ha = lambda l: (PID[l] % 2 == 0); hb = lambda l: (PID[l] % 2 == 1)
        uA = X[t][ha(t), li].mean(0) - X[c][ha(c), li].mean(0); uA = uA / uA.norm()
        uB = X[t][hb(t), li].mean(0) - X[c][hb(c), li].mean(0); uB = uB / uB.norm()
        a = (auc(X[t][hb(t), li] @ uA, X[c][hb(c), li] @ uA) + auc(X[t][ha(t), li] @ uB, X[c][ha(c), li] @ uB)) / 2
        pr = {l: round((X[l][:, li] @ un).mean().item(), 2) for l in X}
        sd = round((X[c][:, li] @ un).std().item(), 2)
        print(f'L{L:2d} |u| {u.norm():6.2f} split-cos {(uA @ uB).item():.2f} heldout-AUC {a:.3f} cos(raw,posmatched) {cosm:.2f} (n {nm}) proj {pr} sd_contrast {sd}')
        D[f'{name}_L{L}'] = un; D[f'{name}m_L{L}'] = um / um.norm() if um is not None else un
    # logit lens at the last layer (final norm, tied head)
    u = D[f'{name}m_L23']; z = E @ (u * (1 + nw))
    print('lens +', [tok.decode([i]) for i in z.topk(25).indices.tolist()])
    print('lens -', [tok.decode([i]) for i in (-z).topk(25).indices.tolist()])
for a in ('enum', 'over'):
    print('cos enum/over by layer', a, [round((D[f'enumm_L{L}'] @ D[f'overm_L{L}']).item(), 2) for L in LAY][:1] if a == 'enum' else '')
print('cos enumm vs overm per layer', [round((D[f'enumm_L{L}'] @ D[f'overm_L{L}']).item(), 2) for L in LAY])
torch.save(D, outd); print('saved', outd)
