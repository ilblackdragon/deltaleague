import torch, time, sys
torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
B='/home/user/delta/exp-base/model'
t0=time.time()
try:
    m=AutoModelForCausalLM.from_pretrained(B, dtype=torch.float32)
except Exception as e:
    print('causal failed', repr(e)[:300]); from transformers import AutoModelForImageTextToText; m=AutoModelForImageTextToText.from_pretrained(B, dtype=torch.float32)
m.eval(); print(type(m).__name__, 'load', time.time()-t0)
print([n for n,_ in m.named_modules() if n.endswith('norm') and 'layers' not in n][:5])
tok=AutoTokenizer.from_pretrained(B)
msgs=[{'role':'user','content':'What is 17*23? Think step by step.'}]
text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=True)
print(repr(text[-200:]))
ids=tok(text,return_tensors='pt').input_ids
print('prompt tokens',ids.shape)
hs={}
norm=dict(m.named_modules())['model.norm']
norm.register_forward_hook(lambda mod,i,o: hs.__setitem__('h',o.detach()))
with torch.no_grad():
    for L in [512, 2048]:
        x=torch.randint(1000,20000,(1,L))
        t0=time.time(); out=m(input_ids=x, use_cache=False); dt=time.time()-t0
        print(L,'tokens forward',round(dt,1),'s', 'h',hs['h'].shape, 'logits', out.logits.shape if out.logits is not None else None)
with torch.no_grad():
    out=m(input_ids=ids,use_cache=False); 
E=m.get_input_embeddings().weight
z=hs['h'][0,-1]@E.T; print('check tied: max|logits-hE|',(out.logits[0,-1]-z).abs().max().item()); print('top5 next',[tok.decode([i]) for i in z.topk(5).indices.tolist()], z.topk(5).values.tolist(), 'mean logit', z.mean().item())
