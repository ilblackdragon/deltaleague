"""Decision points: thinking positions where the leader gives </think> non-trivial probability. Which tokens hold the
rest of the mass there (i.e. what competes with closing)? Split by trace class and by pre/post gold answer."""
import torch, sys, collections, json
sys.argv = [sys.argv[0]] + sys.argv[1:]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
D = torch.load(sys.argv[1]); THID = tok.convert_tokens_to_ids('</think>')
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
mass = collections.defaultdict(collections.Counter); npos = collections.Counter(); pth = collections.Counter()
for t in D['traces']:
    n = t['topi'].shape[0]; end = t['think_end'] if t['think_end'] >= 0 else n; end = min(end + 1, n)
    ti = t['topi'][:end].long(); tv = t['topv'][:end].float()
    top, idx = tv[:, :20], ti[:, :20]; p = torch.softmax(top, 1); cs = p.cumsum(1); p = p * ((cs - p) < 0.95); p = p / p.sum(1, keepdim=True)
    pt = (p * (idx == THID)).sum(1)
    g = 'trunc' if t['truncated'] else ('ok' if t['score'] > 0 else 'bad')
    sel = (pt > thr).nonzero().flatten().tolist()
    npos[g] += len(sel); pth[g] += pt[sel].sum().item(); npos[g + '_all'] += end; pth[g + '_all'] += pt.sum().item()
    for k in sel:
        for j in range(20):
            if p[k, j] > 0: mass[g][idx[k, j].item()] += p[k, j].item()
print('threshold', thr)
for g in ['trunc', 'ok', 'bad']:
    if not npos[g + '_all']: continue
    print(f'\n[{g}] thinking positions {npos[g+"_all"]}, decision points {npos[g]}, sum p(</think>) all={pth[g+"_all"]:.2f} at decision pts={pth[g]:.2f}')
    tot = sum(mass[g].values())
    for i, m in mass[g].most_common(30): print(f'   {tok.convert_ids_to_tokens(i)!r:22} share {m/tot:6.3f}  mass {m:7.2f}')
