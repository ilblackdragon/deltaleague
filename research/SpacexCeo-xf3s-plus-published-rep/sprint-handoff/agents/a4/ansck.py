"""Collateral check for a </think>-row edit (the row is also the INPUT embedding of </think>, tied weights):
teacher-force full generations (prompt + thinking + </think> + answer) under parent and candidate, compare
mean KL(parent||cand) and log-prob of the realised tokens, separately for thinking positions, the first answer
positions, and the rest of the answer; plus p(</think>) at t=0 under both.
usage: ansck.py PARENT CAND TEXTS.jsonl RESULT.json PROMPTS.json N"""
import sys, json, numpy as np, torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
torch.set_num_threads(6); torch.set_grad_enabled(False)
par, cand, tx, rj, pj, N = sys.argv[1:7]; N = int(N)
tok = AutoTokenizer.from_pretrained(par)
M = [Qwen3_5ForCausalLM.from_pretrained(p, dtype=torch.bfloat16).eval() for p in (par, cand)]
rows = {r['id']: r for r in json.load(open(rj))['rows']}; P = {p['id']: p for p in json.load(open(pj))}
T = [json.loads(l) for l in open(tx)]
T = [t for t in T if '</think>' in t['text'] and not rows[t['id']]['truncated']][:N]
acc = {'think': [], 'ans16': [], 'ans': []}; lp = {'think': [[], []], 'ans16': [[], []], 'ans': [[], []]}; p0 = [[], []]
for t in T:
    pre = tok.apply_chat_template([{'role': 'user', 'content': P[t['id']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']; gid = tok(t['text'], add_special_tokens=False)['input_ids'][:6000]
    ids = torch.tensor([pid + gid]); c = gid.index(248069) if 248069 in gid else len(gid)
    L = [torch.log_softmax(m(input_ids=ids).logits[0, len(pid) - 1:-1].float(), -1) for m in M]
    kl = (L[0].exp() * (L[0] - L[1])).sum(-1).numpy()
    tgt = torch.tensor(gid)
    rl = [l.gather(1, tgt[:, None])[:, 0].numpy() for l in L]
    seg = {'think': slice(0, c), 'ans16': slice(c + 1, c + 17), 'ans': slice(c + 17, len(gid))}
    for k, s in seg.items():
        acc[k].append(kl[s]); lp[k][0].append(rl[0][s]); lp[k][1].append(rl[1][s])
    for i in (0, 1): p0[i].append(float(L[i][0, 248069].exp()))
    print(t['id'], 'close at', c, 'KL think %.2e ans16 %.2e ans %.2e' % (kl[seg['think']].mean(), kl[seg['ans16']].mean(), kl[seg['ans']].mean() if len(gid) > c + 17 else 0), flush=True)
for k in acc:
    a = np.concatenate(acc[k]); print(f'{k:6s}: mean KL {a.mean():.2e} (max {a.max():.2e}); mean logprob parent {np.concatenate(lp[k][0]).mean():.4f} cand {np.concatenate(lp[k][1]).mean():.4f}')
print('p(</think>) at t=0: parent mean %.4f max %.4f | cand mean %.6f max %.6f' % (np.mean(p0[0]), np.max(p0[0]), np.mean(p0[1]), np.max(p0[1])))
