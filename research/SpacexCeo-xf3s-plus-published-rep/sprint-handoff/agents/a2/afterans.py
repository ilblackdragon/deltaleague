"""After the gold answer first appears in the thinking (gsm8k/math numeric answers, heuristic match),
what starts the following lines? Truncated vs closed-correct traces."""
import json, sys, re, collections
pre=sys.argv[1]
P={p['id']:p for p in json.load(open(sys.argv[2]))}
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
def pat(a):
    a=str(a).strip().replace('\\!','').replace(',','')
    if re.fullmatch(r'-?\d+(\.\d+)?',a): return re.compile(r'(?:=|is|\\boxed\{)\s*\$?\s*'+re.escape(a)+r'(?![\d.]*\d)')
    return None
first=collections.Counter(); N=collections.Counter(); stats=collections.defaultdict(list)
for t in T:
    r=R[(t['id'],t['sample'])]; p=P[t['id']]
    if p['bench'] not in ('gsm8k','math500'): continue
    rx=pat(p['answer']);
    if not rx: continue
    th=t['text'].split('</think>')[0]; m=rx.search(th)
    g='trunc' if r['truncated'] else ('ok' if r['score']>0 else 'bad')
    stats[g+'_n'].append(1); 
    if not m: continue
    stats[g+'_found'].append(m.start()/max(1,len(th)))
    rest=th[m.end():]
    ntok_after=len(rest)
    stats[g+'_chars_after'].append(ntok_after)
    for ln in rest.split('\n')[1:]:
        ln=re.sub(r'^[\s\*\-\d\.\)#]+','',ln); w=re.findall(r"[A-Za-z']+",ln[:40])[:2]
        if w and ln[:1].isalpha(): first[(g,' '.join(w[:1]))]+=1; first[(g,' '.join(w))]+=1; N[g]+=1
for g in ['trunc','ok','bad']:
    n=len(stats[g+'_n']); f=stats[g+'_found']
    print(g,'traces',n,'gold found in thinking',len(f), 'median rel pos', sorted(f)[len(f)//2] if f else None, 'median chars after', sorted(stats[g+'_chars_after'])[len(f)//2] if f else None, 'lines after',N[g])
keys=set(k for (g,k),_ in first.most_common(300))
rows=[(k,*(1000*first[(g,k)]/max(1,N[g]) for g in ['trunc','ok','bad'])) for k in keys]
print(f"{'line start after gold':24} trunc  ok   bad (per 1k lines)")
for k,a,b,c in sorted(rows,key=lambda r:-r[1])[:45]: print(f'{k:24} {a:6.1f} {b:6.1f} {c:6.1f}')
