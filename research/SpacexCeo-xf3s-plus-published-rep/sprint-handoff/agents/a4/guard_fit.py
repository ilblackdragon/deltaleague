"""Fit the think-start guard with positives = hn at the first generated position for ALL proxy prompts (prompt-only
forward pass), negatives = teacher-forced positions from probe npz files (newlines, natural closes x20, random others).
Held-out check on a prompt split. Saves OUTPREFIX_w.npy (fit on all).
usage: guard_fit.py MODEL_DIR PROMPTS.json NPZ_DIR RESULT.json OUTPREFIX"""
import numpy as np, sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from haz import load
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
torch.set_num_threads(6); torch.set_grad_enabled(False)
mdir, pf, nd, rj, outp = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(mdir)
P = json.load(open(pf))
cache = outp + '_h0.npy'
if os.path.exists(cache): H0 = np.load(cache)
else:
    m = Qwen3_5ForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).eval()
    st = {}; m.model.norm.register_forward_hook(lambda mod, i, o: st.__setitem__('h', o))
    H0 = []
    for p in P:
        pre = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        m(input_ids=torch.tensor([tok(pre, add_special_tokens=False)['input_ids']]), logits_to_keep=1)
        H0.append(st['h'][0, -1].float().numpy())
    H0 = np.array(H0); np.save(cache, H0)
NL = np.array(['\n' in tok.decode([i]) for i in range(len(tok))])
D = load(nd, rj); rng = np.random.default_rng(3)
def negs(sel):
    X, wt = [], []
    for d in sel:
        T = len(d['lc']); h = d['hn'].astype(np.float32); nl = np.zeros(T, bool); nl[1:] = NL[d['gid'][:T - 1]]
        i = np.where(nl & (np.arange(T) >= 16))[0]; X.append(h[i]); wt.append(np.ones(len(i)))
        if T > 16:
            o = rng.choice(np.arange(16, T), size=min(300, T - 16), replace=False); X.append(h[o]); wt.append(np.ones(len(o)))
        if d['closed'] and T > 1: X.append(h[-1:]); wt.append(np.full(1, 20.0))
    return np.concatenate(X), np.concatenate(wt)
def fit(pos, nsel, lam=10.0):
    Xn, wn = negs(nsel)
    X = np.concatenate([pos, Xn]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(Xn))])
    wt = np.concatenate([np.full(len(pos), 20.0 * len(Xn) / max(1, 20 * len(pos)) * 0 + 20.0), wn])
    Xw = X * np.sqrt(wt)[:, None]
    return np.linalg.solve(Xw.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ (y * np.sqrt(wt)))
ids = np.array([p['id'] for p in P]); half = (ids % 2 == 0)
key = lambda d: int(d['key'].split('_')[0])
for lam in [10.0, 100.0]:
    for k, msk in enumerate([half, ~half]):
        w = fit(H0[msk], [d for d in D if (key(d) % 2 == 0) == (k == 0)], lam)
        s0 = H0[~msk] @ w
        ev = [d for d in D if (key(d) % 2 == 0) != (k == 0)]
        Xn, _ = negs(ev); sn = Xn @ w
        sc = np.array([d['hn'][-1].astype(np.float32) @ w for d in ev if d['closed'] and len(d['lc']) > 1])
        print(f'lam {lam:g} fold {k}: heldout t0 s (1,5,50%) {np.round(np.percentile(s0,[1,5,50]),3)} | heldout negatives s (50,99,99.9,max) {np.round(np.percentile(sn,[50,99,99.9,100]),3)} | natural closes (50,max) {np.round([np.median(sc), sc.max()],3) if len(sc) else None}')
w = fit(H0, D, 10.0); np.save(outp + '_w.npy', w)
print('saved', outp + '_w.npy', '|w|', np.linalg.norm(w), 'in-sample t0 s min/median', (H0 @ w).min(), np.median(H0 @ w))
