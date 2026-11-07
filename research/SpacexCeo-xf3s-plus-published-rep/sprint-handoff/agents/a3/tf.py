"""Teacher-force generated traces and measure the vLLM sampler (presence penalty -> temperature 1 -> top-k 20 -> top-p 0.95)
as a function of the final-norm gain g (and optional per-token output-row factors), offline, on CPU.

usage: tf.py MODEL_DIR G0 TRACES.jsonl RESULT.json IDS(comma) OUT.npz [threads]
  MODEL_DIR : checkpoint that generated the traces (its final norm already contains gain G0)
  z = h_normed @ E^T / G0 is the gain-1 logit of that checkpoint (hedge rows / </think> row edits included).
  For each config c=(g, row-factor set) the sampler logit is  l = g * r_v * z_v - 1.5 * [v generated before].
Per position we store, for every config: entropy, nucleus size, log q(actual next token) (-inf if outside nucleus),
p(</think>), p(hedge set), mass on penalized tokens, and (config-independent) whether the actual token was penalized,
raw-logit rank of actual token, fraction of the raw top-20 that is penalized."""
import sys, json, math, os, time
import numpy as np, torch
model_dir, G0, tr_file, res_file, ids, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
torch.set_num_threads(int(sys.argv[7]) if len(sys.argv) > 7 else 6)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(model_dir)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, 'language_model') else m.model
E = m.get_output_embeddings().weight.detach().float()            # tied embedding (V, d)
V = E.shape[0]
voc = tok.get_vocab()
def tid(s): return voc[s.replace(' ', 'Ġ')]
THINK = 248069
HEDGE = [tid(s) for s in [' wait', 'wait', ' Wait', 'Wait', 'Hmm', ' hmm', ' Hmm', ' Maybe', 'Maybe', ' Actually', 'Actually',
                          ' Alternatively', 'Alternatively']]
# token classes for per-token temperature counterfactuals
inv = {i: s for s, i in voc.items()}
DIG = [i for s, i in voc.items() if s.lstrip('Ġ').isdigit() and len(s.lstrip('Ġ')) <= 3]
MATHSYM = [voc[s] for s in ['=', 'Ġ=', '+', 'Ġ+', '-', 'Ġ-', '*', 'Ġ*', '/', 'Ġ/', '^', '(', ')', 'Ġ(', '{', '}', '\\', 'Ġ\\', '.', ',', 'Ġ$', '$']
           if s in voc]
NL = [i for s, i in voc.items() if set(s) <= {'Ċ'} and len(s) > 0]
GS = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0]
MAYBE = HEDGE[7:]
CONF = [('g%.2f' % g, g, None) for g in GS]
for cls, ids_c in [('dig', DIG), ('sym', MATHSYM), ('nl', NL)]:
    for r in [1.05, 0.95]:
        CONF.append(('g1.40_%s%.2f' % (cls, r), 1.4, (ids_c, r)))
for g in [1.3, 1.4, 1.55]:
    for r in [0.95, 1.02, 1.05, 1.15]:
        CONF.append(('g%.2f_think%.2f' % (g, r), g, ([THINK], r)))
    for r in [0.97, 0.94]:
        CONF.append(('g%.2f_maybe%.2f' % (g, r), g, (MAYBE, r)))
CONF.append(('g1.40_think0.87', 1.4, ([THINK], 1 / 1.15)))
CONF.append(('g1.40_nopen', 1.4, 'nopen'))
rowf = []
for name, g, x in CONF:
    f = torch.full((V,), g)
    if isinstance(x, tuple): f[torch.tensor(x[0])] *= x[1]
    rowf.append(f)
rows = {r['id']: r for r in json.load(open(res_file))['rows']}
texts = {json.loads(l)['id']: json.loads(l) for l in open(tr_file)}
P = {p['id']: p for p in json.load(open('/home/clore/dl/proxy400.json' if os.path.exists('/home/clore/dl/proxy400.json') else 'data/proxy400.json'))}
res = {}
for pid in [int(x) for x in ids.split(',')]:
    t0 = time.time()
    p = P[pid]; gen = texts[pid]['text']
    prompt = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pi = tok(prompt, add_special_tokens=False)['input_ids']; gi = tok(gen, add_special_tokens=False)['input_ids']
    x = torch.tensor([pi + gi])
    with torch.no_grad():
        h = lm(input_ids=x).last_hidden_state[0, len(pi) - 1: len(pi) + len(gi) - 1].float()   # predicts gi[t]
    T = len(gi); C = len(CONF)
    st = {k: np.zeros((C, T), np.float32) for k in ['H', 'n', 'lq', 'pthink', 'phedge', 'ppen']}
    pen_act = np.zeros(T, bool); rank = np.zeros(T, np.int32); top20pen = np.zeros(T, np.float32); zthink = np.zeros(T, np.float32)
    zact = np.zeros(T, np.float32); ztop = np.zeros(T, np.float32)
    inexact = np.zeros(T, bool); top3 = np.zeros((T, 3), np.int32); top3p = np.zeros((T, 3), np.float32)
    pen = torch.zeros(V, dtype=torch.bool)
    hedge_t = torch.tensor(HEDGE); gi_t = torch.tensor(gi)
    B = 128
    for s in range(0, T, B):
        hb = h[s:s + B]; b = hb.shape[0]
        z = hb @ E.T / G0                         # (b, V) gain-1 logits
        # penalty masks per position: tokens generated before position t
        pm = torch.zeros((b, V), dtype=torch.bool)
        for j in range(b):
            pm[j] = pen
            pen[gi[s + j]] = True
        a = gi_t[s:s + b]
        za = z.gather(1, a[:, None])[:, 0]
        rank[s:s + b] = (z > za[:, None]).sum(1).numpy()
        zact[s:s + b] = za.numpy(); ztop[s:s + b] = z.max(1).values.numpy(); zthink[s:s + b] = z[:, THINK].numpy()
        pen_act[s:s + b] = pm.gather(1, a[:, None])[:, 0].numpy()
        top20 = z.topk(20, 1).indices
        top20pen[s:s + b] = pm.gather(1, top20).float().mean(1).numpy()
        K = 256
        cz, cix = z.topk(K, 1)
        sp = torch.tensor([THINK] + HEDGE)[None].expand(b, -1)
        dup = (sp[:, :, None] == cix[:, None, :]).any(2)
        szv = z.gather(1, sp).masked_fill(dup, -1e9)
        cz = torch.cat([cz, szv], 1); cix = torch.cat([cix, sp], 1)
        cpm = pm.gather(1, cix).float()
        hit_all = (cix == a[:, None])
        bound = cz[:, K - 1] * max(GS) * 1.2
        for c, (name, g, x) in enumerate(CONF):
            l = cz * rowf[c][cix]
            if x != 'nopen': l = l - 1.5 * cpm
            v, j = l.topk(20, 1); ix = cix.gather(1, j)
            if c == 0: inexact[s:s + b] = (v[:, -1] < bound).numpy()
            pr = torch.softmax(v, 1)
            asc = pr.flip(1); cs = asc.cumsum(1); keep = (cs > 1 - 0.95).flip(1); keep[:, 0] = True
            pr = pr * keep; pr = pr / pr.sum(1, keepdim=True)
            if name == 'g1.40':
                top3[s:s + b] = ix[:, :3].numpy(); top3p[s:s + b] = pr[:, :3].numpy()
            st['H'][c, s:s + b] = (-(pr * torch.log(pr.clamp_min(1e-30))).sum(1)).numpy()
            st['n'][c, s:s + b] = keep.sum(1).numpy()
            hit = (ix == a[:, None]); qa = (pr * hit).sum(1)
            st['lq'][c, s:s + b] = torch.log(qa.clamp_min(1e-30)).numpy()
            st['pthink'][c, s:s + b] = (pr * (ix == THINK)).sum(1).numpy()
            st['phedge'][c, s:s + b] = (pr * torch.isin(ix, hedge_t)).sum(1).numpy()
            st['ppen'][c, s:s + b] = (pr * pm.gather(1, ix)).sum(1).numpy()
    r = rows[pid]
    res[pid] = dict(meta=np.array([p['bench'], r['score'], r['truncated'], T, len(gi)], dtype=object), pen_act=pen_act, rank=rank,
                    top20pen=top20pen, zthink=zthink, zact=zact, ztop=ztop, tokens=np.array(gi, np.int32), inexact=inexact, top3=top3, top3p=top3p, **st)
    print(pid, p['bench'], 'score', r['score'], 'trunc', r['truncated'], 'T', T, 'vllm_tokens', r['tokens'], '%.0fs' % (time.time() - t0), flush=True)
    np.savez_compressed(out, confs=np.array([c[0] for c in CONF]),
                        **{f'{pid}__{k}': v for pid, d in res.items() for k, v in d.items()})
