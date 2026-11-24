"""Teacher-force generated traces and record closing statistics per position.
usage: tf.py MODEL_DIR TEXTS.jsonl PROMPTS.json OUTDIR [--ids 1,2,3] [--edit newleader|none] [--maxlen 8192]
For every generated position t (predicting token t+1) inside the thinking part, records:
  lc   : raw logit of </think> (model as given, gain baked into norm weight)
  top  : top-24 presence-penalised logits of all OTHER tokens (desc), to re-simulate vLLM sampling
         (presence_penalty 1.5 on generated tokens, temperature 1, top_k 20, top_p 0.95) under any close-row edit
  rank : rank of </think> among penalised logits (0 = argmax)
  hn   : final-norm output (fp16), so that any edit e_close += delta gives lc' = lc + hn.delta
Edits are applied in memory; weights on disk are never modified."""
import sys, os, json, argparse, numpy as np, torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('texts'); ap.add_argument('prompts'); ap.add_argument('out')
ap.add_argument('--ids', default=''); ap.add_argument('--edit', default='none'); ap.add_argument('--maxlen', type=int, default=8192)
ap.add_argument('--threads', type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
m = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
E = m.model.embed_tokens.weight
CLOSE = 248069
if a.edit == 'newleader':  # a9340e14: pure scale 0.5 on 7 rows, </think> row x1.3, norm gain 1.4 (from ORIGINAL base)
    with torch.no_grad():
        for s in [' wait', 'wait', ' Wait', 'Wait', 'Hmm', ' hmm', ' Hmm']:
            i = tok.convert_tokens_to_ids(s.replace(' ', 'Ġ')); E[i] = (0.5 * E[i].float()).to(E.dtype)
        E[CLOSE] = (1.3 * E[CLOSE].float()).to(E.dtype)
        m.model.norm.weight.copy_((1.4 * (1 + m.model.norm.weight.float()) - 1).to(E.dtype))
Ef = E.float()
P = {p['id']: p for p in json.load(open(a.prompts))}
want = set(int(x) for x in a.ids.split(',') if x)
hn_store = {}
def hook(mod, inp, out): hn_store['h'] = out
m.model.norm.register_forward_hook(hook)
for line in open(a.texts):
    r = json.loads(line)
    if want and r['id'] not in want: continue
    fn = os.path.join(a.out, f"{r['id']}_{r['sample']}.npz")
    if os.path.exists(fn): continue
    p = P[r['id']]
    pre = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']
    think = r['text'].split('</think>')[0]
    closed = '</think>' in r['text']
    gid = tok(think, add_special_tokens=False)['input_ids'][:a.maxlen]
    ids = torch.tensor([pid + gid])
    with torch.no_grad():
        m.model(input_ids=ids)
        h = hn_store['h'][0, len(pid) - 1: len(pid) + len(gid)].float()  # predicts gid[0..] ; last row predicts next after think
    T = h.shape[0]
    lc = np.zeros(T, np.float32); rank = np.zeros(T, np.int32); top = np.zeros((T, 24), np.float32)
    seen = torch.zeros(Ef.shape[0], dtype=torch.bool)
    pen_rows = []
    for c0 in range(0, T, 256):
        hc = h[c0:c0 + 256]
        L = hc @ Ef.T
        for j in range(L.shape[0]):
            t = c0 + j
            if t > 0: seen[gid[t - 1]] = True  # presence penalty over tokens generated so far
            l = L[j]
            l = l - 1.5 * seen.float()
            lc[t] = L[j, CLOSE].item()
            lcp = l[CLOSE].item()  # close never generated inside thinking -> no penalty
            l[CLOSE] = -1e9
            v = torch.topk(l, 24).values
            top[t] = v.numpy()
            rank[t] = int((v > lcp).sum().item()) if lcp > v[-1] else 24 + int((l > lcp).sum().item() > 24)
    np.savez_compressed(fn, lc=lc, top=top, rank=rank, hn=h.numpy().astype(np.float16), gid=np.array(gid, np.int32),
                        closed=closed, ntok=len(gid))
    print(r['id'], r['sample'], 'T', T, 'closed', closed, 'min rank', rank.min(), flush=True)
