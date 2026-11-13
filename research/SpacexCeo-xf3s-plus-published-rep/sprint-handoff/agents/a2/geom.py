"""Embedding geometry of hedge rows (base model). Writes geom.json."""
import json, torch
from safetensors import safe_open
from transformers import AutoTokenizer
B='/home/user/delta/exp-base/model'
f=safe_open(B+'/model.safetensors','pt')
EMB='model.language_model.embed_tokens.weight'
E=f.get_tensor(EMB).float()
tok=AutoTokenizer.from_pretrained(B); V=tok.get_vocab(); iv={i:s for s,i in V.items()}
mu=E.mean(0)
print('E',E.shape,'|mu|',mu.norm().item(),'mean|e|',E.norm(dim=1).mean().item())
LEADER=['Ġwait','ĠMaybe','ĠWait','Wait','Maybe','ĠActually','ĠAlternatively','Actually','Hmm','ĠHmm','Alternatively']
En=E/E.norm(dim=1,keepdim=True)
out={}
for s in LEADER:
    i=V[s]; c=En@En[i]; top=c.topk(25)
    nb=[(iv[j.item()],round(v.item(),3)) for v,j in zip(top.values,top.indices)][1:]
    out[s]={'id':i,'norm':E[i].norm().item(),'cos_mu':torch.nn.functional.cosine_similarity(E[i],mu,0).item(),'nbrs':nb}
    print(s,i,round(E[i].norm().item(),3),[n for n,_ in nb[:15]])
# candidate surface variants
import re
cands=[]
words=['wait','hmm','maybe','actually','alternatively','but','let','hold','double','verify','check','re','hm','perhaps','however','oh','ok','okay','so','no','yes','wait,','recheck','reconsider','again','another','though','hmmm','ah','oops','first','also','then','now','similarly','still']
for s,i in V.items():
    t=s.replace('Ġ',' ').replace('Ċ','\n')
    core=t.strip().lower().strip(',.!?:;')
    if core in words and len(t.strip())<=14: cands.append((s,i))
print(len(cands))
out['_cands']=sorted([(s,i,E[i].norm().item()) for s,i in cands],key=lambda x:x[0].lower())
json.dump(out,open('/home/user/delta/agents/a2/geom.json','w'),indent=0)
for s,i,n in out['_cands']: print(repr(s),i,round(n,3))
