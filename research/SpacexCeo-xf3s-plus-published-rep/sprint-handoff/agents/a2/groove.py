"""At line starts inside verbatim grooves (b8af1dbc TF): how peaked is the next-token distribution, and what holds the
non-top mass (escape candidates)? Also raw logit of Wait rows (restored x2) relative to the top token."""
import sys, torch, collections
sys.argv=[sys.argv[0], sys.argv[1]]
exec(open('/home/user/delta/agents/a2/wait_restore.py').read().split("print('traces'")[0])
alt=collections.Counter(); p1=[]; gapW=[]; n=0; first=collections.Counter()
for t in traces:
    ti=t['ti']; tv=t['tv'].clone(); m=groove_mask(t)
    ids=t['gid'].tolist(); toks=[tok.decode([i]) for i in ids]
    # use full penalised top40 (undo candidate masking for distribution): rebuild from D
    raw=[x for x in D['traces'] if x['id']==t['id']][0]
    TV=raw['topv'][:t['end']].float(); TI=raw['topi'][:t['end']].long()
    P=torch.softmax(TV[:,:20],1); cs=P.cumsum(1); P=P*((cs-P)<0.95); P=P/P.sum(1,keepdim=True)
    for k in range(1,t['end']):
        if '\n' in toks[k-1] and m[k]:
            n+=1; p1.append(P[k,0].item()); first[tok.convert_ids_to_tokens(TI[k,0].item())]+=1
            for j in range(1,20):
                if P[k,j]>0: alt[tok.convert_ids_to_tokens(TI[k,j].item())]+=P[k,j].item()
            zw=max((t['cz'][k,j]+1.5*t['seen'][k,j]).item()*2 for j in W); gapW.append(TV[k,0].item()-zw)
p1=torch.tensor(p1); g=torch.tensor(gapW)
print('groove line starts',n,'p(top1) quantiles 10/50/90%',[round(p1.quantile(q).item(),3) for q in (0.1,0.5,0.9)],'mean non-top mass',round((1-p1).mean().item(),3))
print('restored-Wait logit gap to top1 (x2 rows) quantiles 10/50/90%',[round(g.quantile(q).item(),1) for q in (0.1,0.5,0.9)])
print('top1 tokens at groove line starts',first.most_common(8))
tot=sum(alt.values()); print('alternative mass',[(s,round(v/n,4)) for s,v in alt.most_common(12)])
