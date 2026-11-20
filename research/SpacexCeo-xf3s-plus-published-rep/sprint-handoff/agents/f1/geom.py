"""Embedding geometry of the edited rows (base weights). Logit of token i = g*(h.e_i); pure scale -> g*s*(h.e_i);
mean-interp -> g*(s*h.e_i + (1-s)*h.mu). This script reports norms, cosines with mu, and the row-vs-mean geometry."""
import torch, json
from safetensors import safe_open
from transformers import AutoTokenizer
B='/home/user/delta/exp-base/model'
tok=AutoTokenizer.from_pretrained(B); V=tok.get_vocab()
f=safe_open(B+'/model.safetensors','pt')
E=f.get_tensor('model.language_model.embed_tokens.weight').float()
w=f.get_tensor('model.language_model.norm.weight').float()
mu=E.mean(0); print('|mu|',mu.norm().item(),'|mu|^2',mu.dot(mu).item(),'mean row norm',E.norm(dim=1).mean().item(),'median',E.norm(dim=1).median().item())
print('(1+w) stats mean/std/min/max',(1+w).mean().item(),(1+w).std().item(),(1+w).min().item(),(1+w).max().item())
names=[' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm',' Maybe','Maybe',' Actually','Actually',' Alternatively','Alternatively','</think>','\n\n','</think>\n\n',' But','But',' Let',' So',' Therefore',' Okay','Okay',' Final',' Answer','**','\n']
for n in names:
    k=n.replace(' ','Ġ').replace('\n','Ċ')
    if k not in V: print(n.encode(),'NOT single token'); continue
    i=V[k]; e=E[i]
    print(f'{n!r:20s} id {i:7d} |e| {e.norm().item():.3f} cos(e,mu) {torch.cosine_similarity(e,mu,dim=0).item():+.3f} e.mu/|mu|^2 {(e.dot(mu)/mu.dot(mu)).item():+.3f} |e-mu| {(e-mu).norm().item():.3f}')
# cosine between hedge rows
H=[V[n.replace(' ','Ġ')] for n in [' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm']]
X=E[H]; C=torch.nn.functional.cosine_similarity(X[:,None],X[None],dim=-1); print('hedge cos matrix\n',C.numpy().round(2))
c=E[V['</think>']]; print('cos(close, hedge rows)',[round(torch.cosine_similarity(c,E[i],dim=0).item(),3) for i in H])
