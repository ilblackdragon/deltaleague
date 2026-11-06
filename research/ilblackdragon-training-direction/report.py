"""Export coverage and summaries without raw traces or official-score claims."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
w=p.parse_args().workspace.resolve()
stages={}
for name,n in [('train_student',16),('train_teacher',16),('heldout_student',8),('heldout_candidate',8),('heldout_teacher',8)]:
    path=w/f'local/{name}.generations.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
    stages[name]={'planned':n,'completed':len(rows),'correct':sum(r['score'] for r in rows),
                  'accuracy_on_completed':sum(r['score'] for r in rows)/len(rows) if rows else None,
                  'truncated':sum(r['truncated'] for r in rows),
                  'examples':[{k:v for k,v in r.items() if k not in ['text','token_ids','loop_positions']} for r in rows]}
report={'scope':'Independent training-split experiment; not official League evaluation','stages':stages,
        'submission_id':None,'official_validation_score':None,'official_private_score':None}
for name in ['pipeline-status','edit','outcome']:
    path=w/f'local/{name}.json'
    if path.exists():report[name]=json.loads(path.read_text())
(w/'local/results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in stages.items()},indent=2))
