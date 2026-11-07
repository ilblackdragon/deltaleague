"""Line-initial word n-grams (first 1-3 words of each thinking line), truncated second half vs closed thinking vs answers."""
import json, sys, re, collections
pre=sys.argv[1]
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
C={g:collections.Counter() for g in ['trunc','ok','bad','ans']}; N=collections.Counter()
def lines(s):
    for ln in s.split('\n'):
        ln=re.sub(r'^[\s\*\-\d\.\)]+','',ln)   # strip bullets/numbering
        w=re.findall(r"[A-Za-z']+|[^\sA-Za-z']",ln)[:3]
        if w and re.match('[A-Za-z]',w[0]): yield w
for t in T:
    r=R[(t['id'],t['sample'])]; th=t['text'].split('</think>')[0]
    if r['truncated']: g='trunc'; th=th[len(th)//2:]
    elif r['closed']: g='ok' if r['score']>0 else 'bad'
    else: continue
    for w in lines(th):
        N[g]+=1
        for k in (1,2,3): C[g][' '.join(w[:k])]+=1
    if r['closed']:
        for w in lines(t['text'].split('</think>')[-1]):
            N['ans']+=1
            for k in (1,2,3): C['ans'][' '.join(w[:k])]+=1
print(dict(N))
keys=set(k for k,_ in (C['trunc']+C['ok']).most_common(400))
rows=[(k,*(1000*C[g][k]/max(1,N[g]) for g in ['trunc','ok','bad','ans'])) for k in keys]
print(f"{'line start':28} trunc   ok    bad   ans  ratio(t/ok)")
for k,a,b,c,d in sorted(rows,key=lambda r:-r[1])[:int(sys.argv[2]) if len(sys.argv)>2 else 60]:
    print(f'{k:28} {a:6.1f} {b:6.1f} {c:6.1f} {d:6.1f} {(a+0.5)/(b+0.5):5.2f}')
