"""Teacher-force cd0b95bb's own proxy generations (GPU, bf16) and save residual-stream outputs of layers LAY at labelled positions.
Labels (trace buckets as in agents/a2/trunc_types.py):
  enum_tail : line starts of 'mathy' lines in the last 40% of thinking of truncated never-closed traces bucketed enumeration_or_computation
  comp_ok   : line starts of mathy lines in the thinking of finished-correct traces (same benches)
  ans_over  : line starts in the answer phase (after </think>) of answer_phase_overflow traces, from answer token 150 on
  ans_ok    : line starts in the answer phase of finished-correct traces
  think_ok  : random thinking positions of finished-correct traces (gate check: how much of each direction lives in normal thinking)
usage: collect_p.py MODEL OUT.pt PREFIX1 PROMPTS1 [PREFIX2 PROMPTS2 ...]   (PREFIX -> PREFIX.json + PREFIX.texts.jsonl)"""
import sys, json, re, random, bisect, collections, time, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
model, out = sys.argv[1:3]; pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))
LAY = list(range(6, 24)); CAP = 40; MAXLEN = 8192 + 1024
RECHK = re.compile(r"^(?:[\*\-\d\.\)#>\s]*)(If|But|Maybe|Perhaps|So|Is it|Is there|Let's check|Let's re|Let me check|Let's verify|Let's look|Let's double|No\b|Or\b|Actually|What if|Why|Double|Hold|Alternatively|However|Wait|Hmm|Did I|Could|Does|Re-)", re.I)
def mathy(l): return (l.count('=') + len(re.findall(r'\d', l))) >= 3 and len(re.findall(r'[A-Za-z]{4,}', l)) <= 4
def lines_of(s):
    off = 0
    for ln in s.split('\n'):
        st = off + (len(ln) - len(ln.lstrip())); off += len(ln) + 1
        if ln.strip(): yield st, ln.strip()
def bucket(txt):
    if '</think>' in txt:
        return 'empty' if len(txt.split('</think>')[0].strip()) < 200 else 'answer_phase_overflow'
    L = [l for l in txt.split('\n') if l.strip()]; tail = L[int(len(L) * 0.6):] if len(L) >= 10 else L
    early = collections.Counter(l.strip() for l in L[:int(len(L) * 0.6)])
    dup = sum(1 for l in tail if early[l.strip()] > 0 and len(l.strip()) > 8) / max(1, len(tail))
    rc = sum(1 for l in tail if RECHK.match(l.strip())) / max(1, len(tail))
    mt = sum(1 for l in tail if mathy(l)) / max(1, len(tail))
    code = sum(1 for l in tail if re.search(r'draw\(|label\(|\bfor\s*\(|;\s*$|\{\s*$|def |return ', l)) / max(1, len(tail))
    if len(L) < 10 or len(txt) / max(1, len(L)) > 800: return 'few_lines'
    if dup >= 0.5: return 'verbatim_loop'
    if code >= 0.3: return 'code'
    if rc >= 0.25: return 'recheck'
    if mt >= 0.5: return 'enumeration_or_computation'
    return 'other'
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
LM = m.model.language_model; cache = {}
def mk(i):
    def h(mod, inp, outp):
        o = outp[0] if isinstance(outp, tuple) else outp
        cache[i] = o[0, cache['pos']].detach().to(torch.float16).cpu()
    return h
for i in LAY: LM.layers[i].register_forward_hook(mk(i))
jobs = []
for pre, pf in pairs:
    P = {p['id']: p for p in json.load(open(pf))}
    R = {(r['id'], r['sample']): r for r in json.load(open(pre + '.json'))['rows']}
    for l in open(pre + '.texts.jsonl'):
        t = json.loads(l); r = R[(t['id'], t['sample'])]; txt = t['text']
        ok = r['score'] == 1 and r['closed'] and not r['truncated']
        b = bucket(txt) if r['truncated'] else ('ok' if ok else 'wrong')
        if b in ('ok', 'enumeration_or_computation', 'answer_phase_overflow') and t['bench'] in ('math500', 'gsm8k', 'mmlu_pro'):
            jobs.append((P[t['id']]['prompt'], txt, b, t['id'], t['sample'], t['bench']))
rng = random.Random(0)
ok_jobs = [j for j in jobs if j[2] == 'ok']; rng.shuffle(ok_jobs)
jobs = [j for j in jobs if j[2] != 'ok'] + [j for j in ok_jobs if j[5] == 'math500'][:160] + [j for j in ok_jobs if j[5] != 'math500'][:80]
print('jobs', collections.Counter(j[2] for j in jobs), flush=True)
recs = []; t0 = time.time()
for n, (prompt, txt, b, pid, smp, bench) in enumerate(jobs):
    pre = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pidz = tok(pre, add_special_tokens=False)['input_ids']
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids = (pidz + enc['input_ids'])[:MAXLEN]; g0 = len(pidz); ng = len(ids) - g0
    starts = [o[0] for o in enc['offset_mapping'][:ng]]
    def posof(off):
        k = bisect.bisect_right(starts, off) - 1
        return g0 + k - 1 if 1 <= k < ng else None
    ci = txt.find('</think>'); think = txt if ci < 0 else txt[:ci]
    lab = []
    if b == 'enumeration_or_computation':
        Ls = list(lines_of(think)); cut = int(len(Ls) * 0.6)
        lab += [(posof(o), 'enum_tail') for o, l in Ls[cut:] if mathy(l)]
    if b == 'ok':
        lab += [(posof(o), 'comp_ok') for o, l in lines_of(think) if mathy(l)]
        a0 = ci + len('</think>')
        lab += [(posof(a0 + o), 'ans_ok') for o, l in lines_of(txt[a0:])]
        tp = [g0 + k for k in range(5, len(starts)) if starts[k] < ci]
        lab += [(p, 'think_ok') for p in rng.sample(tp, min(CAP, len(tp)))]
    if b == 'answer_phase_overflow':
        a0 = ci + len('</think>'); ka = bisect.bisect_left(starts, a0)
        lab += [(p, 'ans_over') for p in (posof(a0 + o) for o, l in lines_of(txt[a0:])) if p is not None and p - g0 >= ka + 150]
    byl = collections.defaultdict(list)
    for p, l in lab:
        if p is not None and p < len(ids) - 1: byl[l].append(p)
    sel = []
    for l, ps in byl.items():
        ps = sorted(set(ps)); sel += [(p, l) for p in rng.sample(ps, min(CAP, len(ps)))]
    if not sel: continue
    sel.sort(); cache['pos'] = torch.tensor([p for p, _ in sel])
    with torch.no_grad(): LM(input_ids=torch.tensor(ids)[None].cuda(), use_cache=False)
    recs.append({'id': pid, 'sample': smp, 'bench': bench, 'bucket': b, 'labels': [l for _, l in sel], 'pos': [p - g0 for p, _ in sel], 'ci_tok': (bisect.bisect_left(starts, ci) if ci >= 0 else -1),
                 'H': torch.stack([cache[i] for i in LAY], 1)})
    if n % 20 == 0: print(n, len(jobs), b, len(ids), {k: len(v) for k, v in byl.items()}, f'{time.time() - t0:.0f}s', flush=True)
torch.save({'layers': LAY, 'recs': recs}, out); print('saved', out, len(recs))
