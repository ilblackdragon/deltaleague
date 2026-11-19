"""Offline selectivity of direction-ablation configs: teacher-force cd0b95bb traces with hooks that apply o <- o - a (o.u_L) u_L on
the writers (self_attn.o_proj / linear_attn.out_proj / mlp.down_proj) of a layer band (equivalent to W <- W - a u u^T W), and
report mean KL(base || edit) and the change of log p(realised token) at each label (ans_over, ans_ok, think_ok, comp_ok, enum_tail).
usage: tf_p.py MODEL DIRS.pt OUT.json N_TRACES_PER_BUCKET PREFIX1 PROMPTS1 [PREFIX2 PROMPTS2]"""
import sys, json, random, bisect, collections, torch
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from transformers import AutoTokenizer, AutoModelForImageTextToText
model, dirs, out, nper = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); pairs = list(zip(sys.argv[5::2], sys.argv[6::2]))
import re
MAXLEN = 8192 + 1024
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
D = torch.load(dirs)
CONF = {
    'over_L12-14_a1': ('over', range(12, 15), 1.0), 'over_L12-14_a2': ('over', range(12, 15), 2.0),
    'over_L10-14_a1': ('over', range(10, 15), 1.0), 'over_L16-18_a1': ('over', range(16, 19), 1.0),
    'over_L20-22_a1': ('over', range(20, 23), 1.0),
    'enumm_L12-14_a1': ('enumm', range(12, 15), 1.0), 'enumm_L16-18_a1': ('enumm', range(16, 19), 1.0),
}
tok = AutoTokenizer.from_pretrained(model)
m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16).cuda().eval(); LM = m.model.language_model
state = {'cfg': None}
def mk(i):
    def h(mod, inp, outp):
        c = state['cfg']
        if c is None or i not in c[1]: return None
        u = D[f'{c[0]}_L{i}'].to(outp.device, torch.float32)
        o = outp.float(); o = o - c[2] * (o @ u).unsqueeze(-1) * u
        return o.to(outp.dtype)
    return h
for i, l in enumerate(LM.layers):
    for part in ('self_attn.o_proj', 'linear_attn.out_proj', 'mlp.down_proj'):
        mod = l; ok = True
        for p in part.split('.'):
            if not hasattr(mod, p): ok = False; break
            mod = getattr(mod, p)
        if ok: mod.register_forward_hook(mk(i))
jobs = collections.defaultdict(list)
for pre, pf in pairs:
    P = {p['id']: p for p in json.load(open(pf))}
    R = {(r['id'], r['sample']): r for r in json.load(open(pre + '.json'))['rows']}
    for l in open(pre + '.texts.jsonl'):
        t = json.loads(l); r = R[(t['id'], t['sample'])]; txt = t['text']
        okr = r['score'] == 1 and r['closed'] and not r['truncated']
        b = bucket(txt) if r['truncated'] else ('ok' if okr else 'wrong')
        if b in ('ok', 'enumeration_or_computation', 'answer_phase_overflow') and t['bench'] in ('math500', 'gsm8k', 'mmlu_pro'):
            jobs[b].append((P[t['id']]['prompt'], txt, b))
rng = random.Random(1); sel_jobs = []
for b, v in jobs.items(): rng.shuffle(v); sel_jobs += v[:nper]
agg = {c: collections.defaultdict(lambda: [0.0, 0.0, 0]) for c in CONF}
for n, (prompt, txt, b) in enumerate(sel_jobs):
    pre = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pz = tok(pre, add_special_tokens=False)['input_ids']; enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids = (pz + enc['input_ids'])[:MAXLEN]; g0 = len(pz); ng = len(ids) - g0; starts = [o[0] for o in enc['offset_mapping'][:ng]]
    ci = txt.find('</think>'); think = txt if ci < 0 else txt[:ci]
    lab = collections.defaultdict(list)
    if ci >= 0:
        ka = bisect.bisect_left(starts, ci + len('</think>'))
        apos = [g0 + k - 1 for k in range(ka + 1, ng)]
        if b == 'answer_phase_overflow': lab['ans_over'] = [p for p in apos if p - g0 >= ka + 150]
        if b == 'ok': lab['ans_ok'] = apos
        tp = [g0 + k - 1 for k in range(5, min(ka, ng))]
    else: tp = [g0 + k - 1 for k in range(5, ng)]
    if b == 'ok': lab['think_ok'] = tp
    if b == 'enumeration_or_computation': lab['enum_tail'] = tp[int(len(tp) * 0.6):]
    lab = {k: sorted(rng.sample(v, min(300, len(v)))) for k, v in lab.items() if v}
    x = torch.tensor(ids)[None].cuda(); tgt = torch.tensor(ids[1:]).cuda()
    allp = torch.tensor(sorted(set(sum(lab.values(), [])))).cuda(); idx = {p: j for j, p in enumerate(allp.tolist())}
    state['cfg'] = None
    with torch.no_grad(): lb = torch.log_softmax(m.lm_head(LM(input_ids=x, use_cache=False).last_hidden_state[0, allp]).float(), -1)
    for c, cfg in CONF.items():
        state['cfg'] = cfg
        with torch.no_grad(): le = torch.log_softmax(m.lm_head(LM(input_ids=x, use_cache=False).last_hidden_state[0, allp]).float(), -1)
        kl = (lb.exp() * (lb - le)).sum(-1); t = tgt[allp]
        dl = le.gather(1, t[:, None])[:, 0] - lb.gather(1, t[:, None])[:, 0]
        for k, ps in lab.items():
            j = torch.tensor([idx[p] for p in ps]).cuda(); a = agg[c][k]; a[0] += kl[j].sum().item(); a[1] += dl[j].sum().item(); a[2] += len(ps)
    state['cfg'] = None
    if n % 10 == 0: print(n, len(sel_jobs), b, len(ids), flush=True)
res = {c: {k: {'kl': round(v[0] / v[2], 4), 'dlogp': round(v[1] / v[2], 4), 'n': v[2]} for k, v in a.items()} for c, a in agg.items()}
json.dump(res, open(out, 'w'), indent=1)
for c, r in res.items(): print(c, {k: (v['kl'], v['dlogp']) for k, v in r.items()})
