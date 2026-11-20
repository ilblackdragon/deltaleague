"""a6: teacher-forced logit-hazard analysis (CPU). Runs a checkpoint over its own proxy generations, stores per-position
candidate logits (base-row logits z0 = h0.e_j with g=1) so that counterfactual specs (hedge scale s, close gain c, norm gain g)
can be evaluated analytically: logit_j = g*z0_j (ordinary), g*s*z0_j (scaled hedge rows), g*c*z0_close; presence penalty 1.5 on
already-generated tokens; top_k 20, top_p 0.95, T=1. First-order only (trajectory held fixed; input-embedding side of the
tied rows not re-run).
usage: tf_hazard.py MODEL BASE TEXTS RESULTS OUT.npz N MAXT GAIN_IN_MODEL"""
import json, sys, time, torch, numpy as np
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(6)
MODEL, BASE, TEXTS, RES, OUT = sys.argv[1:6]; N = int(sys.argv[6]); MAXT = int(sys.argv[7]); GM = float(sys.argv[8])
K = 512
tok = AutoTokenizer.from_pretrained(BASE); V = tok.get_vocab()
H7 = [V[t] for t in ['Ġwait', 'wait', 'ĠWait', 'Wait', 'Hmm', 'Ġhmm', 'ĠHmm']]
HO = [V[t] for t in ['ĠMaybe', 'Maybe', 'ĠActually', 'Actually', 'ĠAlternatively', 'Alternatively']]
CL = V['</think>']; SPEC = H7 + HO + [CL]
E = safe_open(BASE + '/model.safetensors', 'pt').get_tensor('model.language_model.embed_tokens.weight').float()
mu = E.mean(0)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
base = m.model
P = {p['id']: p for p in json.load(open('proxy400.json'))}
rows = {(r['id'], r['sample']): r for r in json.load(open(RES))['rows']}
T = [json.loads(l) for l in open(TEXTS)]
T.sort(key=lambda t: t['id'])
cl = [t for t in T if rows[(t['id'], t['sample'])]['closed']]; tr = [t for t in T if not rows[(t['id'], t['sample'])]['closed']]
sel = cl[::max(1, len(cl) // (N // 2))][:N // 2] + tr[::max(1, len(tr) // (N - N // 2))][:N - N // 2]
acc = {k: [] for k in ['zt', 'st', 'zs', 'ss', 'pos', 'trace', 'think', 'nxt_close', 'closed', 'score', 'zmu']}
meta = []
for ti, t in enumerate(sel):
    r = rows[(t['id'], t['sample'])]
    ptxt = tok.apply_chat_template([{'role': 'user', 'content': P[t['id']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(ptxt, add_special_tokens=False)['input_ids']; gid = tok(t['text'], add_special_tokens=False)['input_ids'][:MAXT]
    ids = torch.tensor([pid + gid]); t0 = time.time()
    with torch.no_grad():
        hs = base(input_ids=ids).last_hidden_state[0].float() / GM
    hs = hs[len(pid) - 1: len(pid) - 1 + len(gid)]           # hidden that predicts gid[i]
    first_close = gid.index(CL) if CL in gid else len(gid)
    seen = np.zeros(E.shape[0], bool)
    for a in range(0, len(gid), 256):
        b = min(len(gid), a + 256)
        with torch.no_grad():
            z = hs[a:b] @ E.T
        acc['zmu'].append((hs[a:b] @ mu).numpy())
        zs = z[:, SPEC].numpy().copy(); z[:, SPEC] = -1e9
        zt, it = torch.topk(z, K, dim=1); zt = zt.numpy(); it = it.numpy()
        st = np.zeros_like(zt, bool); ss = np.zeros_like(zs, bool)
        for i in range(b - a):
            p = a + i
            st[i] = seen[it[i]]; ss[i] = seen[SPEC]
            seen[gid[p]] = True
        acc['zt'].append(zt.astype(np.float16)); acc['st'].append(st); acc['zs'].append(zs.astype(np.float32)); acc['ss'].append(ss)
        acc['pos'].append(np.arange(a, b)); acc['trace'].append(np.full(b - a, ti)); acc['think'].append(np.arange(a, b) < first_close)
        acc['nxt_close'].append(np.array([gid[p] == CL for p in range(a, b)])); acc['closed'].append(np.full(b - a, r['closed'])); acc['score'].append(np.full(b - a, r['score']))
    meta.append((t['id'], t['bench'], r['closed'], r['truncated'], r['score'], len(gid), first_close))
    print(ti, t['id'], t['bench'], 'closed', r['closed'], 'len', len(gid), 'first_close', first_close, f'{time.time()-t0:.0f}s', flush=True)
np.savez_compressed(OUT, **{k: np.concatenate(v) for k, v in acc.items()}, meta=np.array([list(map(float, x[2:])) for x in meta]), meta_id=np.array([x[0] for x in meta]), meta_bench=np.array([x[1] for x in meta]))
print('saved', OUT)
