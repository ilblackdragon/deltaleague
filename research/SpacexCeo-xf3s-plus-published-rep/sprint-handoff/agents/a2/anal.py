"""Offline first-order simulation of extra row edits on teacher-forced traces (tf_box.py output).
Logit model: z_c' = s*z_c_raw (+ (1-s)*h.mu in mean mode) - 1.5*seen_c ; sampling = top_k 20, top_p 0.95, T=1.
Reports expected emissions of each candidate token and of </think> per trace class / phase, before vs after."""
import torch, json, sys, re, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
D = torch.load(sys.argv[1]); cand = D['cand']; cid = D['cid']; C = len(cand)
P = {p['id']: p for p in json.load(open('/home/user/delta/agents/a2/proxy400.json'))}
TH = cand.index('</think>')
def gold_tokpos(t):
    p = P[t['id']]; a = str(p['answer']).strip().replace(',', '')
    if p['bench'] not in ('gsm8k', 'math500') or not re.fullmatch(r'-?\d+(\.\d+)?', a): return None
    rx = re.compile(r'(?:=|is|\\boxed\{)\s*\$?\s*' + re.escape(a) + r'(?![\d.]*\d)')
    pieces = [tok.decode([i]) for i in t['gid'].tolist()]
    cum = 0; s = ''
    text = ''.join(pieces); m = rx.search(text.split('</think>')[0])
    if not m: return -1
    for k, pc in enumerate(pieces):
        cum += len(pc)
        if cum >= m.end(): return k + 1
    return -1
def prep(t):
    gid = t['gid'].long(); n = t['topi'].shape[0]; gid = gid[:n]
    end = t['think_end'] if t['think_end'] >= 0 else n          # thinking positions only (incl. the </think> decision)
    end = min(end + 1, n)
    ti = t['topi'][:end].long(); tv = t['topv'][:end].float(); cz = t['cz'][:end].float(); hmu = t['hmu'][:end].float()
    cidt = torch.tensor(cid)
    seen = torch.zeros(end, C, dtype=torch.bool); s = set()
    gl = gid.tolist()
    for k in range(end):
        if k > 0: s.add(gl[k - 1])
        for j, c in enumerate(cid):
            if c in s: seen[k, j] = True
    # union pool: top40 + candidates (candidate duplicates inside top40 are masked out of top40 part)
    dup = (ti[:, :, None] == cidt[None, None, :]).any(2)
    tv = tv.masked_fill(dup, -1e9)
    t2 = dict(t); t2.update(ti=ti, tv=tv, cz=cz, hmu=hmu, seen=seen.float(), end=end, gid=gid[:end], gpos=gold_tokpos(t))
    return t2
def dist(t, edit, mode='scale'):
    cz = t['cz'].clone(); raw = cz + 1.5 * t['seen']
    for j, s in edit.items():
        nz = s * raw[:, j] + ((1 - s) * t['hmu'] if mode == 'mean' else 0)
        cz[:, j] = nz - 1.5 * t['seen'][:, j]
    v = torch.cat([t['tv'], cz], 1)
    top, idx = v.topk(20, 1); p = torch.softmax(top, 1); cs = p.cumsum(1); p = p * ((cs - p) < 0.95); p = p / p.sum(1, keepdim=True)
    full = torch.zeros_like(v).scatter_(1, idx, p)
    return full[:, 40:]          # probs of candidates
def run(traces, edit, mode='scale'):
    out = collections.defaultdict(lambda: torch.zeros(C))
    for t in traces:
        pc = dist(t, edit, mode)
        g = 'trunc' if t['truncated'] else ('ok' if t['score'] > 0 else 'bad')
        out[g] += pc.sum(0); out[g + '_n'] += t['end']
        gp = t['gpos']
        if gp is not None and gp >= 0:
            out[g + '_postgold'] += pc[gp:].sum(0); out[g + '_postgold_n'] += max(0, t['end'] - gp)
            out[g + '_pregold'] += pc[:gp].sum(0); out[g + '_pregold_n'] += gp
    return out
if __name__ == '__main__':
    traces = [prep(t) for t in D['traces']]
    print('traces', len(traces), collections.Counter('trunc' if t['truncated'] else ('ok' if t['score'] > 0 else 'bad') for t in traces),
          'with gold pos', sum(1 for t in traces if t['gpos'] is not None and t['gpos'] >= 0))
    torch.save(traces, sys.argv[1].replace('.pt', '.prep.pt'))
