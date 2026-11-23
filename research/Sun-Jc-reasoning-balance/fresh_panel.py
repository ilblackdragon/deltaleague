"""Reserve additional MATH500 and BBH items outside the public suite."""
import argparse
import copy
import hashlib
from pathlib import Path

from delta_cli.public_eval.common import digest
from delta_cli.public_eval.suite import manifest
from delta_cli.public_eval.text import load_tasks, verify_tasks
from evaluate import FOCUS, compare_groups, read, run_group, serve, write


def prepare(root):
    full=manifest()
    suite=copy.deepcopy(full)
    suite.update(id='reasoning-balance-fresh-focus',scope='out-of-manifest-target-audit',code={},bfcl={},
                 text={g:suite['text'][g] for g in FOCUS})
    for g,group in suite['text'].items():
        tasks=load_tasks(group['task'],suite['dataset_revisions'])
        verify_tasks(tasks,group['selection'])
        for name,spec in group['selection'].items():
            docs=tasks[name].eval_docs
            pinned=full['text'][g]['selection'][name]
            occupied={digest(docs[i]) for i in pinned['indices']}
            amount=24 if g=='math500' else 2
            unused=set(range(len(docs)))-set(pinned['indices'])
            ranking=sorted(unused,key=lambda i:hashlib.sha256(f'reasoning-balance-panel-v1:{name}:{i}'.encode()).digest())
            chosen=[]
            for i in ranking:
                h=digest(docs[i])
                if h not in occupied:
                    chosen.append(i)
                    occupied.add(h)
                if len(chosen)==amount:
                    break
            assert len(chosen)==amount
            spec['indices']=sorted(chosen)
            spec['doc_sha256']=[digest(docs[i]) for i in spec['indices']]
        verify_tasks(tasks,group['selection'])
    suite.pop('sha256')
    suite['sha256']=digest(suite)
    path=root/'fresh-suite.json'
    if path.exists():
        assert read(path)==suite
    write(path,suite)
    summary={g:sum(len(t['indices']) for t in suite['text'][g]['selection'].values()) for g in FOCUS}
    assert summary=={'math500':24,'bbh':54}
    write(root/'fresh-selection.json',dict(items=summary,suite_sha256=suite['sha256'],
                                          excluded_official_suite_sha256=full['sha256']))


def audit(root):
    chosen=read(root/'replication/selected.json')
    focus=read(root/'replication/focus.json')
    enabled=bool(chosen['tier']) or all(v['candidate_correct']>v['parent_correct'] for v in focus.values())
    record=dict(enabled=enabled,arm=chosen['arm'],suite=read(root/'fresh-selection.json'))
    if enabled:
        arm=chosen['arm']
        dest=root/'fresh-audit'
        dest.mkdir(exist_ok=True)
        suite=read(root/'fresh-suite.json')
        values={'parent':{},arm:{}}
        with serve(root,arm,dest) as (plan,_):
            for g,order in zip(FOCUS,(('parent',arm),(arm,'parent')),strict=True):
                for model in order:
                    values[model].update(run_group(suite,g,model,plan,dest))
        record['scores']=compare_groups(values['parent'],values[arm],FOCUS)
    write(root/'fresh-result.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('action',choices=['prepare','audit'])
    args=parser.parse_args()
    (prepare if args.action=='prepare' else audit)(args.root)
