"""Answer phase (after </think>): does <|im_end|> ever get probability? per-trace max p(im_end), #positions with p>0.01,
logit gap of im_end to top1 (only known when im_end is in the stored top-40)."""
import torch, sys
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
EOS=tok.convert_tokens_to_ids('<|im_end|>'); TH=tok.convert_tokens_to_ids('</think>')
D=torch.load(sys.argv[1])
for t in D['traces']:
    gid=t['gid'].long(); n=t['topi'].shape[0]
    te=t['think_end']
    if te<0: print(t['id'],'no close'); continue
    ti=t['topi'][te+1:n].long(); tv=t['topv'][te+1:n].float()
    p=torch.softmax(tv[:,:20],1); cs=p.cumsum(1); p=p*((cs-p)<0.95); p=p/p.sum(1,keepdim=True)
    pe=(p*(ti[:,:20]==EOS)).sum(1)
    inTop=(ti==EOS); gap=[]
    for k in range(len(ti)):
        w=inTop[k].nonzero().flatten()
        if len(w): gap.append((tv[k,0]-tv[k,w[0]]).item())
    g=torch.tensor(gap) if gap else torch.tensor([float('nan')])
    print(t['id'],'think_end',te,'answer positions',len(ti),'max p(im_end) %.4f'%pe.max().item(),'positions p>0.01:',(pe>0.01).sum().item(),
          '| im_end in top40 at',len(gap),'positions; gap to top1 min %.1f median %.1f'%(g.min().item(),g.median().item()) if gap else '| im_end never in top40')
