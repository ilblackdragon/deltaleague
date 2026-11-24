"""TF the ANSWER phase (after </think>) of traces: store per position the raw logit of EOS (<|im_end|>), the top-8 raw logits
with seen flags (presence penalty set includes thinking tokens), mean logit. usage: f1_tf_answer.py BASE TEXTS RES PROMPTS OUTDIR ids threads"""
import sys, os, json, time, numpy as np, torch
base, texts, resf, prompts, outd, ids_arg = sys.argv[1:7]; thr = int(sys.argv[7]) if len(sys.argv) > 7 else 4
torch.set_num_threads(thr); torch.set_grad_enabled(False); os.makedirs(outd, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(base); voc = tok.get_vocab(); EOS = 248044
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, 'language_model') else m.model
E = m.get_output_embeddings().weight.detach().float(); V = E.shape[0]
P = {p['id']: p for p in json.load(open(prompts))}
rows = {r['id']: r for r in json.load(open(resf))['rows']}
T = [json.loads(l) for l in open(texts)]
want = set(int(x) for x in ids_arg.split(',')); T = [r for r in T if r['id'] in want]
for r in T:
    fn = os.path.join(outd, f"{r['id']}_{r['sample']}.npz")
    if os.path.exists(fn) or '</think>' not in r['text']: continue
    t0 = time.time(); p = P[r['id']]
    pre = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']
    think, ans = r['text'].split('</think>', 1)
    tid = tok(think, add_special_tokens=False)['input_ids'] + [voc['</think>']]
    aid = tok(ans, add_special_tokens=False)['input_ids']
    trunc = rows[r['id']]['truncated']
    if not trunc: aid = aid + [EOS]   # the model must predict EOS itself
    ids = (pid + tid + aid)[:8192 + len(pid)]
    x = torch.tensor([ids[:-1]])
    h = lm(input_ids=x).last_hidden_state[0, len(pid) + len(tid) - 1:].float()   # row j predicts aid[j]
    n = h.shape[0]; aid = aid[:n]
    seen = torch.zeros(V, dtype=torch.bool); seen[torch.tensor(tid)] = True
    ze = np.zeros(n, np.float32); topv = np.zeros((n, 8), np.float32); topi = np.zeros((n, 8), np.int32); tseen = np.zeros((n, 8), bool); eseen = np.zeros(n, bool)
    for s in range(0, n, 128):
        z = h[s:s + 128] @ E.T; tv, ti = torch.topk(z, 8, dim=1)
        topv[s:s + 128] = tv.numpy(); topi[s:s + 128] = ti.numpy(); ze[s:s + 128] = z[:, EOS].numpy()
        for j in range(z.shape[0]):
            t = s + j
            if t > 0: seen[aid[t - 1]] = True
            tseen[t] = seen[ti[j]].numpy(); eseen[t] = bool(seen[EOS])
    np.savez_compressed(fn, ze=ze, topv=topv, topi=topi, tseen=tseen, eseen=eseen, aid=np.array(aid, np.int32), trunc=trunc, nthink=len(tid))
    print(r['id'], 'answer tokens', n, 'trunc', trunc, f'{time.time()-t0:.0f}s', flush=True)
