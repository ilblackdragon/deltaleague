"""One teacher-forced pass per trace that records (a) closing statistics at every position and (b) a forced-close
answer probe every STEP tokens ("budget forcing": append '</think>\n\nThe final answer is \\boxed{' and decode greedily).
usage: probe.py MODEL_DIR TEXTS.jsonl PROMPTS.json OUTDIR --ids a,b,c [--step 512] [--threads 6]
Per trace writes OUTDIR/<id>_<sample>.npz with
  lc[t]   raw </think> logit predicting token t (t=0 .. T, last = position right after the thinking text)
  top[t]  top-24 presence-penalised logits of the other tokens (presence_penalty 1.5 over generated tokens)
  hn[t]   final-norm output (fp16)
  gid     generated token ids, probe_pos, probe_ans (decoded strings), probe_ok (bool)"""
import sys, os, json, copy, argparse, numpy as np, torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('texts'); ap.add_argument('prompts'); ap.add_argument('out')
ap.add_argument('--ids', default=''); ap.add_argument('--step', type=int, default=512); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--noprobe', action='store_true')
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
m = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
Ef = m.model.embed_tokens.weight.float()
CLOSE = 248069
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
LET = [tok.convert_tokens_to_ids(c) for c in 'ABCDEFGHIJ']
P = {p['id']: p for p in json.load(open(a.prompts))}
want = [x for x in a.ids.split(',') if x]
store = {}
m.model.norm.register_forward_hook(lambda mod, i, o: store.__setitem__('h', o))
PROBE = tok('\n</think>\n\nThe final answer is $\\boxed{', add_special_tokens=False)['input_ids']

def run(ids, cache):
    o = m(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
    return o.logits[0, -1].float(), o.past_key_values

def probe(cache, p):
    c = copy.deepcopy(cache)
    lg, c = run(PROBE, c)
    if p['scorer'] == 'mc':
        L = lg[LET]; return 'ABCDEFGHIJ'[int(L.argmax())], float(torch.softmax(L, 0).max())
    out = []; depth = 1
    for _ in range(20):
        t = int(lg.argmax()); s = tok.decode([t])
        depth += s.count('{') - s.count('}')
        if depth <= 0:
            out.append(t); break
        out.append(t); lg, c = run([t], c)
    s = tok.decode(out)
    if depth <= 0: s = s[:s.rfind('}')]
    return s, 0.0

def ok(p, s):
    if p['scorer'] == 'mc': return s == p['answer']
    try:
        g = parse(str(p['answer']), extraction_config=CFG); pr = parse('$' + s + '$', extraction_config=CFG)
        return bool(g) and bool(pr) and bool(verify(g, pr))
    except Exception: return False

T_ = {}
for line in open(a.texts):
    r = json.loads(line)
    k = f"{r['id']}_{r['sample']}"
    if k in want or str(r['id']) in want: T_[k] = r
for k in (want if want else list(T_)):
    if k not in T_:
        k2 = [x for x in T_ if x.split('_')[0] == k]
        if not k2: continue
        k = k2[0]
    r = T_[k]; fn = os.path.join(a.out, k + '.npz')
    if os.path.exists(fn): continue
    p = P[r['id']]
    pre = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']
    think = r['text'].split('</think>')[0]
    gid = tok(think, add_special_tokens=False)['input_ids'][:8192]
    # prompt pass (keeps last-position hidden for predicting gid[0])
    cache = None
    o = m(input_ids=torch.tensor([pid]), use_cache=True, logits_to_keep=1); cache = o.past_key_values
    H = [store['h'][0, -1:].float()]
    ppos, pans, pok, pconf = [], [], [], []
    st_ = 1024 if a.noprobe else (256 if ('</think>' in r['text'] and len(gid) < 3000) else a.step)
    bounds = list(range(0, len(gid), st_)) + [len(gid)]
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        o = m(input_ids=torch.tensor([gid[s0:s1]]), past_key_values=cache, use_cache=True, logits_to_keep=1); cache = o.past_key_values
        H.append(store['h'][0].float())
        ans, conf = ('', 0.0) if a.noprobe else probe(cache, p)
        ppos.append(s1); pans.append(ans); pok.append(ok(p, ans)); pconf.append(conf)
    h = torch.cat(H, 0)[:len(gid) + 1]
    T = h.shape[0]
    lc = np.zeros(T, np.float32); top = np.zeros((T, 24), np.float32)
    seen = torch.zeros(Ef.shape[0], dtype=torch.bool)
    gidt = torch.tensor(gid + [0], dtype=torch.long)
    for c0 in range(0, T, 256):
        L = h[c0:c0 + 256] @ Ef.T
        n = L.shape[0]
        lc[c0:c0 + n] = L[:, CLOSE].numpy()
        base = L - 1.5 * seen.float()[None]          # penalty for tokens generated before this chunk
        base[:, CLOSE] = -1e9
        cv, ci = torch.topk(base, 24 + 256, dim=1)
        # tokens generated inside the chunk before position t (t=c0+j sees gid[:t])
        new = gidt[max(0, c0 - 1):c0 + n - 1] if c0 > 0 else torch.cat([torch.tensor([-1]), gidt[:n - 1]])
        for j in range(n):
            nj = new[:j + 1] if c0 > 0 else new[1:j + 1]
            pen = torch.isin(ci[j], nj) & ~seen[ci[j]]
            v = cv[j] - 1.5 * pen.float()
            top[c0 + j] = torch.topk(v, 24).values.numpy()
        seen[gidt[max(0, c0 - 1):c0 + n - 1] if c0 > 0 else gidt[:n - 1]] = True
    np.savez_compressed(fn, lc=lc, top=top, hn=h.numpy().astype(np.float16), gid=np.array(gid, np.int32),
                        probe_pos=np.array(ppos), probe_ans=np.array(pans), probe_ok=np.array(pok), probe_conf=np.array(pconf),
                        closed='</think>' in r['text'], score=-1)
    print(k, p['bench'], 'T', T, 'closed', '</think>' in r['text'], 'probe_ok', ''.join('1' if x else '0' for x in pok), flush=True)
