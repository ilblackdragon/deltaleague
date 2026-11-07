"""Bench-controlled over-representation: per bench, rate of line-initial words per 1k thinking lines in truncated
(2nd half) vs closed-correct traces; pooled with Mantel-Haenszel-like weighting (sum over benches)."""
import json, sys, re, collections
pre=sys.argv[1]
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
W=sys.argv[2].split(',')
cnt=collections.defaultdict(collections.Counter); N=collections.Counter()
for t in T:
    r=R[(t['id'],t['sample'])]; th=t['text'].split('</think>')[0]
    if r['truncated']: g='trunc'; th=th[len(th)//2:]
    elif r['score']>0: g='ok'
    else: continue
    for ln in th.split('\n'):
        ln=re.sub(r'^[\s\*\-\d\.\)#>]+','',ln); w=re.findall(r"[A-Za-z']+",ln[:30])
        if not w or not ln[:1].isalpha(): continue
        N[(t['bench'],g)]+=1; cnt[(t['bench'],g)][w[0]]+=1
        if len(w)>1: cnt[(t['bench'],g)][w[0]+' '+w[1]]+=1
benches=sorted(set(b for b,_ in N))
print('lines',dict(N))
print(f"{'word':16}"+''.join(f'{b:>16}' for b in benches)+'   pooled(t/ok)')
for w in W:
    row=''; num=den=0
    for b in benches:
        a=1000*cnt[(b,'trunc')][w]/max(1,N[(b,'trunc')]); o=1000*cnt[(b,'ok')][w]/max(1,N[(b,'ok')])
        row+=f'{a:7.1f}/{o:<7.1f} '
        wt=min(N[(b,'trunc')],N[(b,'ok')]); num+=wt*a; den+=wt*o
    print(f'{w:16}{row}  {num/max(den,1e-9):5.2f}')
