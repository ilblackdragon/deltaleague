"""Classify truncated generations: verbatim loop (tail lines repeat earlier lines) vs open-ended re-checking."""
import json, sys, collections, re
pre=sys.argv[1]
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
res=collections.Counter(); per=collections.defaultdict(list)
hed=re.compile(r'\b(Wait|wait|Hmm|hmm|Maybe|maybe|Actually|actually|Alternatively|alternatively|But wait)\b')
for t in T:
    r=R[(t['id'],t['sample'])]
    if not r['truncated']: continue
    L=[l.strip() for l in t['text'].split('\n') if len(l.strip())>8]
    if len(L)<20: res['few_lines']+=1; per[t['bench']].append(1.0); continue
    q=len(L)*3//4; early=collections.Counter(L[:q]); tail=L[q:]
    dup=sum(1 for l in tail if early[l]>0)/len(tail)
    per[t['bench']].append(dup)
    res['loop>=0.5' if dup>=0.5 else ('mixed0.2-0.5' if dup>=0.2 else 'open<0.2')]+=1
print(pre, dict(res))
for b,v in per.items(): print(b, len(v), 'median dup', sorted(v)[len(v)//2], 'share>=0.5', sum(x>=0.5 for x in v)/len(v))
