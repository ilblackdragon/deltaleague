"""Diff embedding rows + final norm of versions vs baseline (CPU). usage: rowdiff.py VERSION_ID..."""
import sys, torch, json
from safetensors import safe_open
from transformers import AutoTokenizer
B='versions/0ea62f40-7cd8-4d98-9ebb-484192fa05eb/model'
tok=AutoTokenizer.from_pretrained(B); inv={v:k for k,v in tok.get_vocab().items()}
a=safe_open(B+'/model.safetensors','pt')
EMB='model.language_model.embed_tokens.weight'; NORM='model.language_model.norm.weight'
E0=a.get_tensor(EMB).float(); mu=E0.mean(0); N0=a.get_tensor(NORM).float()
for v in sys.argv[1:]:
    b=safe_open(f'versions/{v}/model/model.safetensors','pt')
    E=b.get_tensor(EMB).float(); d=(E-E0).abs().sum(1); rows=(d>0).nonzero().flatten().tolist()
    out=[]
    for r in rows:
        # fit E[r] = s*E0[r] + c*mu by least squares
        X=torch.stack([E0[r],mu],1); coef=torch.linalg.lstsq(X,E[r].unsqueeze(1)).solution.flatten().tolist()
        out.append((r,inv[r],round(coef[0],3),round(coef[1],3)))
    N=b.get_tensor(NORM).float(); gr=((1+N)/(1+N0)); 
    other=[k for k in b.keys() if k not in (EMB,NORM) and not torch.equal(a.get_tensor(k),b.get_tensor(k))]
    print(v, 'rows', out, 'gain mean/std', round(gr.mean().item(),4), round(gr.std().item(),4), 'other changed', other[:10])
