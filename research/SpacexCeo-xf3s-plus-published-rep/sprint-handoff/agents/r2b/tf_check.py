import sys, json, torch, numpy as np
sys.path.insert(0, '/home/user/delta/agents/a5/box')
from common import load_model, lm
tok, m = load_model('/home/user/delta/exp-xf3s/model', threads=4)
D = json.load(open('/home/user/delta/agents/r2b/diag_rows.json'))['ref']
P = {p['id']: p for p in json.load(open('/home/user/delta/agents/a1/proxy400.json'))}
T = {}
for l in open('/home/user/delta/agents/a1/res/ref_xf3s_x4.texts.jsonl'):
    t = json.loads(l); T[(t['id'], t['sample'])] = t['text']
pick = []
for bb, n in (('gsm8k', 3), ('math500', 2), ('mmlu_pro', 2)):
    rs = sorted([r for r in D if r['src'] == 'ref_xf3s_x4.json' and r['bench'] == bb and r['healthy'] and 300 < r['tokens'] < 1100], key=lambda r: (r['pid'], r['sample']))
    pick += rs[:n]
E = lm(m).embed_tokens.weight.detach()
def edited(groups):
    E2 = E.clone()
    for ids, s in groups:
        ids = torch.tensor(ids); X = E2[ids]; mu = X.mean(0, keepdim=True); E2[ids] = mu + s * (X - mu)
    return E2
DIG = list(range(15, 25)); LET = list(range(32, 42))
heads = {'dig115': edited([(DIG, 1.15)]), 'dl115': edited([(DIG, 1.15), (LET, 1.15)])}
def samp(z):  # top-k 20 then top-p .95 at T=1 (no presence penalty)
    v, i = z.topk(20, -1); p = v.softmax(-1); c = p.cumsum(-1); keep = (c - p) < 0.95
    p = p * keep; p = p / p.sum(-1, keepdim=True); out = torch.zeros_like(z); out.scatter_(-1, i, p); return out
agg = {k: {'dig_pos': [], 'dig_p_before': [], 'dig_p_after': [], 'non_tv': [], 'non_dig_mass_delta': [], 'let_pos_p': []} for k in heads}
for r in pick:
    pre = tok.apply_chat_template([{'role': 'user', 'content': P[r['pid']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    a = tok(pre, add_special_tokens=False)['input_ids']; g = tok(T[(r['id'], r['sample'])], add_special_tokens=False)['input_ids']
    ids = torch.tensor(a + g)[None]
    with torch.no_grad():
        h = lm(m)(input_ids=ids, use_cache=False).last_hidden_state[0]  # after final norm
    h = h[len(a) - 1:-1]; nxt = torch.tensor(g)
    z0 = h @ E.T; p0 = samp(z0)
    for k, E2 in heads.items():
        z1 = h @ E2.T; p1 = samp(z1)
        isd = torch.isin(nxt, torch.tensor(DIG)); isl = torch.isin(nxt, torch.tensor(LET))
        A = agg[k]
        A['dig_p_before'] += p0[isd, nxt[isd]].tolist(); A['dig_p_after'] += p1[isd, nxt[isd]].tolist()
        nd = ~isd & ~isl
        A['non_tv'] += (0.5 * (p1[nd] - p0[nd]).abs().sum(-1)).tolist()
        A['non_dig_mass_delta'] += (p1[nd][:, DIG].sum(-1) - p0[nd][:, DIG].sum(-1)).tolist()
        if isl.any(): A['let_pos_p'] += list(zip(p0[isl, nxt[isl]].tolist(), p1[isl, nxt[isl]].tolist()))
    print(r['bench'], r['pid'], len(g), flush=True)
for k, A in agg.items():
    b, a_ = np.array(A['dig_p_before']), np.array(A['dig_p_after']); tv = np.array(A['non_tv']); dm = np.array(A['non_dig_mass_delta'])
    print(f"{k}: digit positions n={len(b)}: mean p(actual) {b.mean():.4f}->{a_.mean():.4f}; share p<0.9 {np.mean(b<.9):.3f}->{np.mean(a_<.9):.3f}; mean(1-p) {np.mean(1-b):.4f}->{np.mean(1-a_):.4f}")
    print(f"   non-digit positions n={len(tv)}: mean TV {tv.mean():.5f}, max {tv.max():.3f}, share TV>0.05 {np.mean(tv>.05):.4f}; digit mass delta mean {dm.mean():+.5f} max {dm.max():+.3f}")
    if A['let_pos_p']:
        L = np.array(A['let_pos_p']); print(f"   letter positions n={len(L)}: p(actual) {L[:,0].mean():.4f}->{L[:,1].mean():.4f}")
