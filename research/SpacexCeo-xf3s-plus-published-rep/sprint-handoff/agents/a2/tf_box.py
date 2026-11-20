"""Teacher-force generations through a model on CPU (GPU box, 6 threads) and dump, per generated position:
top-40 penalised logits (ids, vals), raw logits of a candidate token set, h.mu (gain included), emitted token.
usage: tf_box.py MODEL_DIR TEXTS.jsonl RESULT.json PROMPTS.json OUT.pt [max_traces] [max_len]"""
import sys, json, torch, time, random
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
torch.set_num_threads(6)
model, texts, resf, pf, outf = sys.argv[1:6]
maxn = int(sys.argv[6]) if len(sys.argv) > 6 else 40
maxlen = int(sys.argv[7]) if len(sys.argv) > 7 else 8192
tok = AutoTokenizer.from_pretrained(model)
m = Qwen3_5ForConditionalGeneration.from_pretrained(model, dtype=torch.float32).eval()
lm = m.model.language_model; E = lm.embed_tokens.weight.data
BASE_E = None
V = tok.get_vocab()
CAND = json.load(open(sys.argv[0].replace('tf_box.py', 'cand.json')))
cid = [V.get(s.replace(' ', 'Ġ').replace('\n', 'Ċ'), tok.convert_tokens_to_ids(s)) for s in CAND]
P = {p['id']: p for p in json.load(open(pf))}
R = {(r['id'], r['sample']): r for r in json.load(open(resf))['rows']}
T = [json.loads(l) for l in open(texts)]
random.seed(0)
tr = [t for t in T if R[(t['id'], t['sample'])]['truncated']]; cl = [t for t in T if R[(t['id'], t['sample'])]['closed']]
random.shuffle(tr); random.shuffle(cl)
sel = [x for pair in zip(tr[:maxn // 2], cl[:maxn - maxn // 2]) for x in pair]
import os
if os.environ.get('IDS'):
    want = [int(x) for x in os.environ['IDS'].split(',')]; byid = {t['id']: t for t in T}; sel = [byid[i] for i in want]
mu_base = torch.load(sys.argv[0].replace('tf_box.py', 'mu_base.pt'))   # mean of ORIGINAL embedding rows
out = []
t0 = time.time()
for t in sel:
    r = R[(t['id'], t['sample'])]
    ptxt = tok.apply_chat_template([{'role': 'user', 'content': P[t['id']]['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(ptxt, add_special_tokens=False)['input_ids']
    gid = tok(t['text'], add_special_tokens=False)['input_ids'][:maxlen]
    if os.environ.get('CLOSEONLY'):
        TID = tok.convert_tokens_to_ids('</think>')
        if TID in gid: gid = gid[:gid.index(TID) + 1]
    ids = pid + gid
    with torch.no_grad():
        h = lm(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, len(pid) - 1:len(ids) - 1]   # predicts gid[k]
        n = h.shape[0]; topv = []; topi = []; cz = []; hmu = (h @ mu_base)
        seen = torch.zeros(E.shape[0], dtype=torch.bool)
        for a in range(0, n, 256):
            z = h[a:a + 256] @ E.T
            for k in range(z.shape[0]):
                if a + k > 0: seen[gid[a + k - 1]] = True
                z[k][seen] -= 1.5
            v, i = z.topk(40, dim=1); topv.append(v.half()); topi.append(i.int()); cz.append(z[:, cid].half())
    out.append({'id': t['id'], 'bench': t['bench'], 'truncated': r['truncated'], 'closed': r['closed'], 'score': r['score'],
                'gid': torch.tensor(gid, dtype=torch.int32), 'think_end': (gid.index(V['</think>']) if '</think>' in V and V['</think>'] in gid else (gid.index(tok.convert_tokens_to_ids('</think>')) if tok.convert_tokens_to_ids('</think>') in gid else -1)),
                'topv': torch.cat(topv), 'topi': torch.cat(topi), 'cz': torch.cat(cz), 'hmu': hmu.half()})
    print(len(out), t['id'], n, f'{time.time() - t0:.0f}s', flush=True)
    torch.save({'cand': CAND, 'cid': cid, 'traces': out}, outf)
