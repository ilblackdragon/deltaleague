# Build a local copy of the new leader a9340e14 from the original baseline (CPU), exactly per the diffed spec.
import torch, os, shutil, hashlib, sys
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
base='/home/user/delta/exp-base/model'; out='/home/user/delta/agents/a5/nl_model'
for f in os.listdir(base):
    if f!='model.safetensors': shutil.copy(os.path.join(base,f), os.path.join(out,f))
W=load_file(os.path.join(base,'model.safetensors'))
v=AutoTokenizer.from_pretrained(base).get_vocab()
EMB='model.language_model.embed_tokens.weight'; NORM='model.language_model.norm.weight'
E=W[EMB].float()
for s in [' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm']: E[v[s.replace(' ','Ġ')]]*=0.5
E[v['</think>']]*=1.3
W[EMB]=E.to(torch.bfloat16)
W[NORM]=(1.4*(1+W[NORM].float())-1).to(torch.bfloat16)
for k in [EMB,NORM]: print(k, hashlib.sha256(W[k].view(torch.uint16).numpy().tobytes()).hexdigest())
save_file(W, os.path.join(out,'model.safetensors'), metadata={'format':'pt'})
