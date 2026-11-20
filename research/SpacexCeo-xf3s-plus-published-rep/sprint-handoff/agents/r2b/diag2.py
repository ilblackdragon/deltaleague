import json, re, numpy as np
D = json.load(open('/home/user/delta/agents/r2b/diag_rows.json'))
NUM = re.compile(r'-?\d[\d,]*\.?\d*')
def lastnum(s):
    m = NUM.findall(s or ''); return m[-1].replace(',', '').rstrip('.') if m else None
def boxed(s):
    b = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', s or ''); return b[-1] if b else None
for k in ('ref', 'h05', 'x10'):
    rows = [r for r in D[k] if r['healthy']]
    by = {}
    for r in rows: by.setdefault(r['pid'], []).append(r)
    # within-prompt paired: prompts with both correct and wrong healthy gens
    dt, da, n = [], [], 0
    for pid, rs in by.items():
        c = [r for r in rs if r['score'] > 0]; w = [r for r in rs if r['score'] == 0]
        if c and w:
            n += 1; dt.append(np.log(np.mean([r['think_tok'] for r in w])) - np.log(np.mean([r['think_tok'] for r in c])))
            da.append(np.log(np.mean([r['ans_tok'] for r in w])) - np.log(np.mean([r['ans_tok'] for r in c])))
    print(f'{k}: within-prompt (n={n}) log(think_w/think_c) {np.mean(dt):+.3f}±{np.std(dt)/np.sqrt(n):.3f}  log(ans_w/ans_c) {np.mean(da):+.3f}±{np.std(da)/np.sqrt(n):.3f}')
    for bb in ('gsm8k', 'math500'):
        rb = [r for r in rows if r['bench'] == bb]
        # does the answer-phase result agree with the thinking's last number?
        agree = [lastnum(r['think_tail']) == lastnum(boxed(r['ans_text'])) for r in rb if boxed(r['ans_text'])]
        ca = [r['score'] for r in rb if boxed(r['ans_text']) and lastnum(r['think_tail']) == lastnum(boxed(r['ans_text']))]
        cd = [r['score'] for r in rb if boxed(r['ans_text']) and lastnum(r['think_tail']) != lastnum(boxed(r['ans_text']))]
        # think ends with "</think>" right after a stated answer vs mid-reasoning
        print(f'   {bb}: ans==think-last-number {100*np.mean(agree):.0f}%  acc agree {100*np.mean(ca):.1f} (n={len(ca)})  acc disagree {100*np.mean(cd):.1f} (n={len(cd)})')
    for bb in ('mmlu_pro', 'arc_c'):
        rb = [r for r in rows if r['bench'] == bb]
        L = lambda s: (re.findall(r'\b([A-J])\b', s or '') or [None])[-1]
        ag = [r['score'] for r in rb if r['letter'] and L(r['think_tail'][-300:]) == r['letter']]
        dg = [r['score'] for r in rb if r['letter'] and L(r['think_tail'][-300:]) != r['letter']]
        print(f'   {bb}: letter==think-last-letter {100*len(ag)/max(1,len(ag)+len(dg)):.0f}%  acc agree {100*np.mean(ag):.1f} (n={len(ag)}) disagree {100*np.mean(dg):.1f} (n={len(dg)})')
# flips on normally-closed prompts: ref strict - h05 strict
P = {}
for k in D:
    for r in D[k]: P.setdefault(k, {}).setdefault(r['pid'], []).append(r)
st = lambda rs: np.mean([r['healthy'] and r['score'] > 0 for r in rs])
hs = lambda rs: np.mean([r['healthy'] for r in rs])
nc = [i for i in P['ref'] if hs(P['ref'][i]) >= .75]
for k in ('h05', 'x10'):
    d = sorted([(st(P[k][i]) - st(P['ref'][i]), i) for i in nc])
    neg = [x for x in d if x[0] <= -1/3]; pos = [x for x in d if x[0] >= 1/3]
    print(f'{k}: normally-closed prompts with strict drop>=1/3: {len(neg)}, gain>=1/3: {len(pos)}; sum d {sum(x[0] for x in d):+.1f}')
    from collections import Counter
    print('   drops by bench', Counter(P['ref'][i][0]['bench'] for _, i in neg), 'gains', Counter(P['ref'][i][0]['bench'] for _, i in pos))
    # on drop prompts: think length ref vs cand (healthy), wrong-gen type
    tr = [np.median([r['think_tok'] for r in P['ref'][i] if r['healthy']]) for _, i in neg]
    tc = [np.median([r['think_tok'] for r in P[k][i] if r['healthy']] or [np.nan]) for _, i in neg]
    print(f'   drop prompts median think ref {np.nanmedian(tr):.0f} -> {k} {np.nanmedian(tc):.0f}; ratio median {np.nanmedian(np.array(tc)/np.array(tr)):.2f}')
    tr = [np.median([r['think_tok'] for r in P['ref'][i] if r['healthy']]) for _, i in pos]
    tc = [np.median([r['think_tok'] for r in P[k][i] if r['healthy']] or [np.nan]) for _, i in pos]
    print(f'   gain prompts median think ref {np.nanmedian(tr):.0f} -> {k} {np.nanmedian(tc):.0f}; ratio median {np.nanmedian(np.array(tc)/np.array(tr)):.2f}')
    allr = [np.median([r['think_tok'] for r in P[k][i] if r['healthy']] or [np.nan]) / np.median([r['think_tok'] for r in P['ref'][i] if r['healthy']]) for i in nc]
    print(f'   all normally-closed think ratio median {np.nanmedian(allr):.2f}')
    print('   worst drops:', [(P['ref'][i][0]['bench'], i, round(x, 2)) for x, i in d[:8]])
