"""Freeze training/check partitions before inference; no League validation access."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from datasets import load_dataset

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
w=p.parse_args().workspace.resolve()
plan=json.loads((w/'plan.json').read_text())
assert not (w/'local/selected.json').exists(), 'Never reseed a started experiment'
ds=load_dataset(plan['dataset'],'main',split='train',revision=plan['dataset_revision'],cache_dir=str(w/'data-cache'))
indices=list(range(len(ds)));random.Random(plan['split_seed']).shuffle(indices)
rows=[];seen=set()
for i in indices:
    d=ds[i]
    normalized=' '.join(d['question'].split())
    if normalized in seen:continue
    seen.add(normalized)
    rows.append({'id':f'gsm8k-train-{i}','dataset_index':i,
                 'split':'training' if len(rows)<16 else 'heldout',
                 'doc_sha256':hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                 'prompt':d['question']+'\nPlease reason step by step and put your final answer in \\boxed{}.',
                 'answer':d['answer'].split('####')[-1].strip(),'scorer':'math'})
    if len(rows)==24:break
assert len(rows)==24 and len({r['id'] for r in rows})==24
path=w/'local/selected.json';path.write_text(json.dumps(rows,indent=2)+'\n')
plan['selected_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
plan['status']='data_frozen'
plan['demonstrations_used']=0
plan['demonstration_exclusion']='No demonstrations used in this independent zero-shot experiment. Official worker demonstration identities unavailable; no claim of exclusion from unknown official few-shot draws.'
(w/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(w/'local/selection-summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ['prompt','answer']} for r in rows],indent=2)+'\n')
print(json.dumps({'training':16,'heldout':8,'selected_sha256':plan['selected_sha256']}))
