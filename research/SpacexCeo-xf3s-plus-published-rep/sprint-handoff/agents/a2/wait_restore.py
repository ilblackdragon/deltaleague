"""On b8af1dbc (Wait rows at 0.5 of original), simulate restoring the 4 Wait rows to s_total of original:
z_Wait' = raw * s_total/0.5. Report expected Wait emissions per 1k positions in (i) groove region (position inside a line
that verbatim-repeats an earlier line) vs (ii) non-groove region, thinking only."""
import sys, torch, collections
sys.argv=[sys.argv[0], sys.argv[1]]
exec(open('/home/user/delta/agents/a2/anal.py').read().split("if __name__")[0])
traces=[prep(t) for t in D['traces']]
W=[cand.index(s) for s in [' Wait','Wait',' wait','wait']]
def groove_mask(t):
    ids=t['gid'].tolist(); toks=[tok.decode([i]) for i in ids]
    mask=torch.zeros(len(ids),dtype=torch.bool); seen=set(); line=[]; start=0
    for k,s in enumerate(toks):
        line.append(s)
        if '\n' in s:
            l=''.join(line).strip()
            if len(l)>8 and l in seen: mask[start:k+1]=True
            seen.add(l); line=[]; start=k+1
    return mask
print('traces',[(t['id'],t['truncated'],t['end']) for t in traces])
for st in [0.5,0.8,0.9,0.95,1.0]:
    agg=collections.Counter()
    for t in traces:
        p=dist(t,{j:st/0.5 for j in W}); pw=p[:,W].sum(1); m=groove_mask(t)[:len(pw)]
        agg['g']+=pw[m].sum().item(); agg['gn']+=m.sum().item(); agg['n']+=pw[~m].sum().item(); agg['nn']+=(~m).sum().item()
    print(f's_total={st:4}: E[Wait] per 1k tokens  groove {1000*agg["g"]/max(1,agg["gn"]):6.2f} (n={agg["gn"]})   non-groove {1000*agg["n"]/max(1,agg["nn"]):6.2f} (n={agg["nn"]})')
