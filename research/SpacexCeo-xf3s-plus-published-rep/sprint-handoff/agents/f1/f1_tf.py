"""F1: teacher-force generated traces through the ORIGINAL base model and store what is needed to recompute the
exact vLLM sampler (presence 1.5 on generated tokens -> T=1 -> top-k 20 -> top-p 0.95) under ANY edit of the family
  z'_i = g * (s_i z_i + (1-s_i) m_t)   [mean mode]   or   g * s_i z_i  [scale]   or   g*c*z_close,  + bias b*m_t/|mu|^2
from the base logits z_i = h_t . e_i. Per position t (predicting generated token t) we store:
  top64 raw logits + ids (covers every token that can be in the penalised top-20 for g>=1), seen flags,
  raw logits of a watch list (hedge rows, </think>, substitutes), mean logit m_t = h.mu, the actual token, and h (fp16).
usage: f1_tf.py BASE_MODEL TEXTS.jsonl PROMPTS.json OUTDIR ids(comma or 'all') [threads] [maxlen]"""
import sys, os, json, time, numpy as np, torch
base, texts, prompts, outd = sys.argv[1:5]
ids_arg = sys.argv[5]; thr = int(sys.argv[6]) if len(sys.argv) > 6 else 4; maxlen = int(sys.argv[7]) if len(sys.argv) > 7 else 8192
torch.set_num_threads(thr); torch.set_grad_enabled(False); os.makedirs(outd, exist_ok=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(base); voc = tok.get_vocab()
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
lm = m.model.language_model if hasattr(m.model, 'language_model') else m.model
E = m.get_output_embeddings().weight.detach().float(); V = E.shape[0]; mu = E.mean(0)
WATCH = [' wait', 'wait', ' Wait', 'Wait', 'Hmm', ' hmm', ' Hmm', ' Maybe', 'Maybe', ' Actually', 'Actually', ' Alternatively', 'Alternatively',
         '</think>', ' But', 'But', ' Let', ' So', ' Therefore', ' Okay', 'Okay', ' However', ' Hold', ' Double', ' Let me', ' Oh', ' Alright', 'Alright',
         ' Final', ' Answer', ' Thus', ' Hence', '\n\n', '\n', ' Yes', ' No', ' Actually', ' Perhaps', ' Oh', ' Then', ' Now', ' Since']
WID = []; WNAME = []
for s in WATCH:
    k = s.replace(' ', 'Ġ').replace('\n', 'Ċ')
    if k in voc and voc[k] not in WID: WID.append(voc[k]); WNAME.append(s)
WID_t = torch.tensor(WID)
P = {p['id']: p for p in json.load(open(prompts))}
T = [json.loads(l) for l in open(texts)]
if ids_arg != 'all':
    want = set(int(x) for x in ids_arg.split(',')); T = [r for r in T if r['id'] in want]
json.dump({'watch': WNAME, 'wid': WID, 'mu_sq': mu.dot(mu).item()}, open(os.path.join(outd, 'meta.json'), 'w'))
for r in T:
    fn = os.path.join(outd, f"{r['id']}_{r['sample']}.npz")
    if os.path.exists(fn): continue
    t0 = time.time(); p = P[r['id']]
    pre = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']
    closed = '</think>' in r['text']
    think = r['text'].split('</think>')[0]
    gid = tok(think, add_special_tokens=False)['input_ids'][:maxlen]
    if closed: gid = gid + [voc['</think>']]           # the model has to predict the close token itself
    x = torch.tensor([pid + gid[:-1]])                  # h at position len(pid)-1+t predicts gid[t]
    h = lm(input_ids=x).last_hidden_state[0, len(pid) - 1:].float()   # (T, d), row t predicts gid[t]
    n = h.shape[0]; K = 64
    topv = np.zeros((n, K), np.float32); topi = np.zeros((n, K), np.int32); topseen = np.zeros((n, K), bool)
    wz = np.zeros((n, len(WID)), np.float32); wseen = np.zeros((n, len(WID)), bool); mlog = np.zeros(n, np.float32)
    seen = torch.zeros(V, dtype=torch.bool); B = 128
    for s in range(0, n, B):
        z = h[s:s + B] @ E.T
        mlog[s:s + B] = (h[s:s + B] @ mu).numpy()
        tv, ti = torch.topk(z, K, dim=1)
        topv[s:s + B] = tv.numpy(); topi[s:s + B] = ti.numpy(); wz[s:s + B] = z[:, WID_t].numpy()
        for j in range(z.shape[0]):
            t = s + j
            if t > 0: seen[gid[t - 1]] = True
            topseen[t] = seen[ti[j]].numpy(); wseen[t] = seen[WID_t].numpy()
    np.savez_compressed(fn, topv=topv, topi=topi, topseen=topseen, wz=wz, wseen=wseen, mlog=mlog, gid=np.array(gid, np.int32),
                        h=h.numpy().astype(np.float16), closed=closed, npre=len(pid), bench=p['bench'], score=0.0)
    print(r['id'], r['sample'], p['bench'], 'n', n, 'closed', closed, f'{time.time()-t0:.0f}s', flush=True)
