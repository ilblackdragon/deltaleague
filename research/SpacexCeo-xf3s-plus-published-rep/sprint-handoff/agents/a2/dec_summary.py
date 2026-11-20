"""Per-trace decision-point summary (p(</think>)>thr under the a9340e14 leader) + competitor tokens."""
import torch, sys, collections
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model'); TH=tok.convert_tokens_to_ids('</think>')
D=torch.load(sys.argv[1]); thr=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
comp=collections.Counter(); rows=[]
for t in D['traces']:
    n=t['topi'].shape[0]; end=t['think_end'] if t['think_end']>=0 else n; end=min(end+1,n)
    ti=t['topi'][:end].long(); tv=t['topv'][:end].float()
    p=torch.softmax(tv[:,:20],1); cs=p.cumsum(1); p=p*((cs-p)<0.95); p=p/p.sum(1,keepdim=True)
    pt=(p*(ti[:,:20]==TH)).sum(1); idx=(pt>thr).nonzero().flatten().tolist()
    pts=[]
    for k in idx:
        others=[(tok.convert_ids_to_tokens(ti[k,j].item()),p[k,j].item()) for j in range(20) if ti[k,j]!=TH and p[k,j]>0]
        for s,q in others: comp[s]+=q
        pts.append(f'{k}:{pt[k]:.2f}')
    cls='trunc' if t['truncated'] else 'closed'
    # P(no close at any decision point) under first-order independence
    pnc=torch.prod(1-pt[:end-1] if t['think_end']>=0 else 1-pt).item()
    rows.append((t['id'],t['bench'],cls,t['score'],t['think_end'],pts,pnc))
for r in rows: print(r[0],r[1],r[2],'score',r[3],'think_end',r[4],'decision pts',r[5],'P(skip all earlier pts)=%.2f'%r[6])
tot=sum(comp.values()); print('competitor mass at decision points:',[(s,round(q/tot,3)) for s,q in comp.most_common(8)])
