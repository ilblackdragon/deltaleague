"""Part (a): on BASE-model traces teacher-forced through the BASE model, how much probability does each edit remove from
the hedge tokens? Compares old leader (mean mode s=0.6, gain 1.4) vs new leader (pure scale 0.5, gain 1.4).
Gain is applied as z -> 1.4 z for all tokens before the presence penalty (penalty already in cz/topv: undo + redo)."""
import sys, torch, collections, json
sys.argv=[sys.argv[0], sys.argv[1]]
exec(open('/home/user/delta/agents/a2/anal.py').read().split("if __name__")[0])
traces=[prep(t) for t in D['traces']]
def gain(t,g):
    t=dict(t); seen=t['seen']
    # topv: penalised values; approximate the penalty on top-40 by: positions where token seen -> unknown; we rescale raw = v (+1.5 if seen) only for candidates exactly;
    # for non-candidates assume penalty was applied (most frequent tokens are seen early): v' = g*(v+1.5)-1.5
    t['tv']=g*(t['tv']+1.5)-1.5; t['cz']=g*(t['cz']+1.5*seen)-1.5*seen; t['hmu']=g*t['hmu']
    return t
G=[gain(t,1.4) for t in traces]
OLD=['Ġwait','ĠMaybe','ĠWait','Wait','Maybe','ĠActually','ĠAlternatively','Actually','Hmm','ĠHmm','Alternatively']
OLD=[s.replace('Ġ',' ') for s in OLD]
NEW=[' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm']
b=run(G,{}); o=run(G,{cand.index(s):0.6 for s in OLD},'mean'); n=run(G,{cand.index(s):0.5 for s in NEW})
print('base traces',len(G),collections.Counter('trunc' if t['truncated'] else 'closed' for t in G),'positions',sum(t['end'] for t in G))
emitted=collections.Counter()
for t in G:
    for i in t['gid'].tolist(): emitted[i]+=1
print(f"{'token':16} emitted  E[count] gain1.4 | old leader (mean .6) | new leader (scale .5)")
for s in [' Wait','Wait',' wait',' Hmm','Hmm',' hmm',' Maybe','Maybe',' maybe',' Actually','Actually',' Alternatively','But',' But','If',' If','So',' So',' Let','</think>']:
    j=cand.index(s); tot=lambda r: sum(r[g][j].item() for g in ('trunc','ok','bad') if g in r)
    print(f'{s!r:16} {emitted[cid[j]]:6d}   {tot(b):8.1f}        | {tot(o):8.2f}            | {tot(n):8.2f}')
