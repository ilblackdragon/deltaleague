"""Heuristic taxonomy of truncated generations (tail = last 40% of the thinking, or the answer for answer-phase overflow).
Order of rules: empty-think -> answer overflow -> verbatim loop -> re-check-driven -> enumeration/computation -> other."""
import json, sys, re, collections
pre=sys.argv[1]
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
RECHK=re.compile(r"^(?:[\*\-\d\.\)#>\s]*)(If|But|Maybe|Perhaps|So|Is it|Is there|Let's check|Let's re|Let me check|Let's verify|Let's look|Let's double|No\b|Or\b|Actually|What if|Why|Double|Hold|Alternatively|However|Wait|Hmm|Did I|Could|Does|Re-)",re.I)
def lines(s): return [l for l in s.split('\n') if l.strip()]
cat=collections.Counter(); byb=collections.defaultdict(collections.Counter); ex={}
for t in T:
    r=R[(t['id'],t['sample'])]
    if not r['truncated']: continue
    txt=t['text']
    if '</think>' in txt:
        think, ans = txt.split('</think>')[0], txt.split('</think>')[-1]
        c='empty_think_then_answer_overflow' if len(think.strip())<200 else 'answer_phase_overflow'
    else:
        L=lines(txt); tail=L[int(len(L)*0.6):] if len(L)>=10 else L
        early=collections.Counter(l.strip() for l in L[:int(len(L)*0.6)])
        dup=sum(1 for l in tail if early[l.strip()]>0 and len(l.strip())>8)/max(1,len(tail))
        rc=sum(1 for l in tail if RECHK.match(l.strip()))/max(1,len(tail))
        mathy=sum(1 for l in tail if (l.count('=')+len(re.findall(r'\d',l)))>=3 and len(re.findall(r'[A-Za-z]{4,}',l))<=4)/max(1,len(tail))
        code=sum(1 for l in tail if re.search(r'draw\(|label\(|\bfor\s*\(|;\s*$|\{\s*$|def |return ',l))/max(1,len(tail))
        if len(L)<10 or len(txt)/max(1,len(L))>800: c='few_lines_long_paragraph'
        elif dup>=0.5: c='verbatim_loop'
        elif code>=0.3: c='code_or_asy'
        elif rc>=0.25: c='recheck_driven'
        elif mathy>=0.5: c='enumeration_or_computation'
        else: c='other_open_ended'
        t['_m']=(round(dup,2),round(rc,2),round(mathy,2))
    cat[c]+=1; byb[t['bench']][c]+=1; ex.setdefault(c,[]).append(t['id'])
n=sum(cat.values()); print(pre,'truncated',n)
for c,k in cat.most_common(): print(f'  {c:34} {k:4d}  {100*k/n:5.1f}%   e.g. ids {ex[c][:6]}')
for b,v in byb.items(): print('  ',b,dict(v))
