import sys, torch
from safetensors import safe_open
a=safe_open(sys.argv[1],'pt'); b=safe_open(sys.argv[2],'pt')
for k in a.keys():
    x=a.get_tensor(k); y=b.get_tensor(k)
    if x.shape!=y.shape or not torch.equal(x,y):
        d=(y.float()-x.float())
        nz=(d!=0)
        print(k, tuple(x.shape), x.dtype, 'changed elems', int(nz.sum()))
        if x.dim()==2:
            rows=nz.any(1).nonzero().flatten().tolist()
            print('  rows changed', len(rows), rows[:20])
            for r in rows[:12]:
                ratio=(y[r].float()/x[r].float().where(x[r]!=0, torch.ones(1)))
                print('   row',r,'median ratio', ratio.median().item())
        else:
            print('  (1+y)/(1+x) median', ((1+y.float())/(1+x.float())).median().item(), 'y/x', (y.float()/x.float()).median().item())
