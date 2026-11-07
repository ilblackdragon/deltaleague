# Analyse the published sparse down_proj edit: rank structure of the delta, column-space in the residual stream.
import json, numpy as np, torch
from safetensors import safe_open
man=json.load(open('upstream/manifest.json'))
f=safe_open('/home/user/delta/exp-base/model/model.safetensors','pt')
res={}
with np.load('upstream/edit.npz',allow_pickle=False) as ed:
  for i,e in enumerate(man['tensors']):
    o=f.get_tensor(e['name']); bits=o.view(torch.uint16).numpy().copy().reshape(-1)
    bits[ed[f't{i}_idx']]=ed[f't{i}_val']
    new=torch.from_numpy(bits.reshape(o.shape)).view(torch.bfloat16).float(); D=(new-o.float())
    cols=torch.nonzero(D.abs().sum(0)>0).flatten()
    U,S,Vh=torch.linalg.svd(D,full_matrices=False)
    e2=(S**2).cumsum(0)/(S**2).sum()
    print(e['name'],'|D|',D.norm().item(),'|W|',o.float().norm().item(),'ncols',len(cols),'S[:8]',S[:8].numpy().round(3),'energy r1,r2,r4,r8',[round(e2[k].item(),3) for k in [0,1,3,7]])
    # how the changed columns relate: per column, ratio of new vs old
    Wc=o.float()[:,cols]; Dc=D[:,cols]
    cosc=(Wc*Dc).sum(0)/(Wc.norm(dim=0)*Dc.norm(dim=0)+1e-9)
    print('  per-column cos(W_col, D_col): mean',cosc.mean().item(),'min',cosc.min().item(),'max',cosc.max().item(),' |D_col|/|W_col| mean',(Dc.norm(dim=0)/Wc.norm(dim=0)).mean().item())
    res[e['name']]={'U':U[:,:16].numpy(),'S':S[:16].numpy(),'cols':cols.numpy()}
np.savez('up_svd.npz',**{f'{k.split(".")[3]}_U':v['U'] for k,v in res.items()},**{f'{k.split(".")[3]}_S':v['S'] for k,v in res.items()},**{f'{k.split(".")[3]}_cols':v['cols'] for k,v in res.items()})
# readout of top singular directions through unembedding (logit lens)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
E=f.get_tensor('model.language_model.embed_tokens.weight').float(); nw=1+f.get_tensor('model.language_model.norm.weight').float()
for k,v in res.items():
  for j in range(3):
    u=torch.tensor(v['U'][:,j]); lg=E@(u*nw)
    for sgn in [1,-1]:
      top=torch.topk(sgn*lg,15).indices
      print(k.split('.')[3],'sv',j,'+' if sgn>0 else '-',[tok.decode([t]) for t in top])
