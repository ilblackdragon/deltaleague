"""Batched mini-eval for Delta League candidates on CPU.
Uses the competition's generation config (temp/top_p/top_k/presence penalty, thinking on) and the CLI's
official score_answer(); only max_new_tokens is overridden. Advisory, NOT the server protocol.
Usage: python mini_eval.py --workspace qwen35 [--weights edited.safetensors] --max-new-tokens 1536 --batch 8 --out x.jsonl
"""
import argparse, json, time, sys, re
from pathlib import Path
import torch
from delta_cli.local import read_workspace, model_directory, Runner
from delta_cli.scoring import score_answer
from transformers import LogitsProcessor, LogitsProcessorList, set_seed

ap=argparse.ArgumentParser(); ap.add_argument('--workspace',default='qwen35'); ap.add_argument('--weights')
ap.add_argument('--items',default='mini_items.json'); ap.add_argument('--max-new-tokens',type=int,default=1536)
ap.add_argument('--batch',type=int,default=8); ap.add_argument('--limit',type=int); ap.add_argument('--out',required=True)
ap.add_argument('--seed',type=int,default=1000)
a=ap.parse_args()
torch.set_num_threads(8)
ws,identity,config,meta=read_workspace(a.workspace)
gen=config['generation']
items=json.load(open(a.items))[:a.limit]
FILLER=[13784,13428,77264,85152,11158,3655,84485]  # Wait, ' Wait', Hmm, ' Hmm', wait, ' wait', ' hmm'
out=open(a.out,'x')
with model_directory(ws,meta,weights=a.weights) as (path,prov):
    r=Runner(path,'cpu'); tok=r.tokenizer; tok.padding_side='left'
    out.write(json.dumps({'meta':True,'provenance':prov,'runner':r.details,'max_new_tokens':a.max_new_tokens,'generation':gen,'seed':a.seed})+'\n')
    pen=gen.get('presence_penalty',1.5)
    for b in range(0,len(items),a.batch):
        chunk=items[b:b+a.batch]
        texts=[tok.apply_chat_template([{'role':'user','content':it['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=gen.get('enable_thinking',True)) for it in chunk]
        inp=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False)
        L=inp['input_ids'].shape[1]
        class PP(LogitsProcessor):
            def __call__(self,ids,scores):
                for row in range(ids.shape[0]):
                    scores[row,ids[row,L:].unique()]-=pen
                return scores
        set_seed(a.seed+b); t0=time.monotonic()
        with torch.inference_mode():
            g=r.model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=gen['temperature'],top_p=gen['top_p'],top_k=gen['top_k'],
                               logits_processor=LogitsProcessorList([PP()]),pad_token_id=tok.pad_token_id)
        dt=time.monotonic()-t0
        for it,row in zip(chunk,g[:,L:]):
            ids=row.tolist()
            eos=[i for i,x in enumerate(ids) if x in (tok.eos_token_id,tok.pad_token_id) or x==248046]
            n=eos[0]+1 if eos else len(ids)
            ids=ids[:n]
            text=tok.decode(ids,skip_special_tokens=False)
            for t in [tok.eos_token,tok.pad_token]:
                if t: text=text.replace(t,'')
            s=score_answer(it,text)
            rec={'id':it['id'],'bench':it['bench'],'score':s,'tokens':n,'truncated':not eos,'closed_think':'</think>' in text,
                 'filler':sum(ids.count(f) for f in FILLER),'text':text}
            out.write(json.dumps(rec)+'\n'); out.flush()
            print(it['id'],s,n,'TRUNC' if not eos else '', rec['filler'],flush=True)
        print(f'batch {b//a.batch} {dt:.0f}s',flush=True)
