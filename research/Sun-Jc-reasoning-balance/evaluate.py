"""Compare all edits with the pinned native scorer and a verified leader."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from delta_cli.public_eval.common import check_versions, digest
from delta_cli.public_eval.suite import PRIMARY
from delta_cli.public_eval.text import run as text_run
from delta_cli.public_eval.coding import run as code_run
from delta_cli.public_eval.tools import run as tool_run

BASE=Path('/Users/jc/delta/worktrees/grpo-delta/research/Sun-Jc-grpo-delta-calibration')
sys.path.insert(0,str(BASE))
from analyze_screen import analyze, expected_ids
from baseline_reuse import verified_baseline, indexed

ARMS=('query-sharp','query-soft','seed-three')
FOCUS=('math500','bbh')
PORT=8787
ENDPOINT=f'http://127.0.0.1:{PORT}'
SERVER_HASH='b06db6d2d7c1be1a8a81f92638b5aab91b7071a1ed4ecfc60f4341040d00652b'
SUITE_HASH='b22f0530da84f9f92d11cae15883270fc3e54d6c1f080e1db2f210ca19c73e00'


def read(p):
    return json.loads(p.read_text())


def write(p,data):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(p)


def note(root,stage,arm,group):
    entry=dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,arm=arm,group=group)
    write(root/'progress.json',entry)
    print(json.dumps(entry),flush=True)


@contextmanager
def serve(root,arm,dest):
    plan_path=root/arm/'plan.json'
    plan=read(plan_path)
    mlx=root.parent.parent/'public-eval/.venv-mlx/bin/python'
    with (dest/'server.log').open('a') as log:
        process=subprocess.Popen([str(mlx),str(BASE/'serve_models_batched8.py'),'--plan',str(plan_path),'--port',str(PORT)],
                                 stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError('Model server exited during startup')
                try:
                    with urllib.request.urlopen(ENDPOINT+'/health',timeout=2) as response:
                        health=json.load(response)
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Model server')
            assert health['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
            assert health['server_sha256']==SERVER_HASH and health['execution']['max_batch_size']==8
            assert health['identities'][arm]['patch_sha256']==plan['candidates'][arm]['patch_sha256']
            assert health['identities']['parent']['official_content_hash']==plan['parent_content_hash']
            assert health['generation_seed']==plan['generation']['seed'] and health['dtype']=='float32'
            old=dest/'backend.json'
            if old.exists():
                assert read(old)==health
            write(old,health)
            yield plan,health
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def rows_from_report(suite,group,path):
    if group in suite['text']:
        raw=read(path)
        metric,chosen_filter=PRIMARY[group]
        values=[]
        for task,spec in suite['text'][group]['selection'].items():
            ids=dict(zip(spec['indices'],spec['doc_sha256'],strict=True))
            samples=[s for s in raw['samples'][task] if s['filter']==chosen_filter]
            assert sorted(s['doc_id'] for s in samples)==sorted(ids)
            for s in samples:
                assert digest(s['doc'])==ids[s['doc_id']]
                values.append(dict(id=f"{task}:{s['doc_id']}",score=float(s[metric])))
        return {group:values}
    if group in suite['code']:
        rows=read(path)['samples']
        return {group:[dict(id=str(r['id']),score=float(r['passed'])) for r in rows]}
    groups={'bfcl_single_turn':[],'bfcl_multi_turn':[]}
    for category,spec in suite['bfcl'].items():
        paths=list(path.with_suffix('').rglob(f'BFCL_v4_{category}_score.json'))
        assert len(paths)==1
        entries=[json.loads(s) for s in paths[0].read_text().splitlines() if s.strip()]
        failed={r['id'] for r in entries[1:]}
        assert failed<=set(spec['ids']) and entries[0]['correct_count']==spec['count']-len(failed)
        group_name='bfcl_multi_turn' if category.startswith('multi_turn_') else 'bfcl_single_turn'
        groups[group_name].extend(dict(id=i,score=float(i not in failed)) for i in spec['ids'])
    return groups


def run_group(suite,group,model,plan,dest):
    directory=dest/model
    directory.mkdir(exist_ok=True)
    path=directory/f'{group}.json'
    receipt=directory/f'{group}.verified.json'
    if receipt.exists():
        old=read(receipt)
        assert old['suite_sha256']==suite['sha256'] and old['report_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        args=dict(concurrency=8,generation=plan['generation'])
        if group in suite['text']:
            text_run(suite,group,ENDPOINT+'/v1',model,plan['parent'],path,**args)
        elif group in suite['code']:
            code_run(suite,group,ENDPOINT+'/v1',model,'',path,sandbox_backend='wasi',**args)
        else:
            tool_run(suite,ENDPOINT+'/v1',model,plan['parent'],path,**args)
    rows=rows_from_report(suite,group,path)
    wanted=expected_ids(suite)
    indexed(rows,{g:wanted[g] for g in rows})
    write(receipt,dict(suite_sha256=suite['sha256'],report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def order(suite):
    return [*FOCUS,*(g for g in suite['text'] if g not in FOCUS),*suite['code'],'bfcl']


def compare_groups(parent,child,keys):
    result={}
    for g in keys:
        a={r['id']:r['score'] for r in parent[g]}
        b={r['id']:r['score'] for r in child[g]}
        assert a.keys()==b.keys()
        result[g]=dict(items=len(a),parent_correct=sum(a.values()),candidate_correct=sum(b.values()),
                       wins=sum(b[k]>a[k] for k in a),losses=sum(b[k]<a[k] for k in a))
    return result


def full_report(dest,suite,parent,child,arm):
    report=analyze({'parent':parent,arm:child},expected_ids(suite))
    report['suite_sha256']=suite['sha256']
    report['focus']=compare_groups(parent,child,FOCUS)
    write(dest/'groups.json',{'parent':parent,arm:child})
    write(dest/'analysis.json',report)
    return report


def evaluate(root, arms=ARMS):
    check_versions()
    source=root.parent/'leader-round2-v1/general'
    suite=read(source/'suite.json')
    assert suite['sha256']==SUITE_HASH
    summaries={}
    for arm in arms:
        dest=root/arm/'development'
        dest.mkdir(parents=True,exist_ok=True)
        if not (root/arm/'validation.log').is_file():
            raise RuntimeError(f'Candidate needs validation: {arm}')
        with serve(root,arm,dest) as (plan,health):
            assert health['execution']==read(source/'backend.json')['execution']
            parent,provenance=verified_baseline(source,suite,plan['generation'],health,BASE/'native_screen_batched8.py')
            write(dest/'baseline.json',provenance)
            child={}
            for group in order(suite):
                note(root,'development',arm,group)
                child.update(run_group(suite,group,arm,plan,dest))
                if all(g in child for g in FOCUS):
                    write(dest/'focus.json',compare_groups(parent,child,FOCUS))
            report=full_report(dest,suite,parent,child,arm)
            summaries[arm]={'macro':report['macro_scores'][arm],'parent':report['macro_scores']['parent'],
                            'delta':report['paired_vs_parent'][arm]['delta_points'],'focus':report['focus'],
                            'interval90':report['paired_vs_parent'][arm]['central_90_interval_points']}
            write(root/'development-results.json',summaries)

    def priority(arm):
        row=summaries[arm]
        gains=[(r['candidate_correct']-r['parent_correct'])/r['items'] for r in row['focus'].values()]
        tier=2 if min(gains)>0 and row['delta']>0 else int(min(gains)>=0 and row['delta']>0)
        return tier,row['delta'],min(gains),-arms.index(arm)

    arm=max(arms,key=priority)
    tier=priority(arm)[0]
    new_suite=read(root.parent/'ten-hour-confirmation-v1/suite.json')
    development_ids=expected_ids(suite)
    for g,ids in expected_ids(new_suite).items():
        assert development_ids[g].isdisjoint(ids)
    dest=root/'replication'
    dest.mkdir(exist_ok=True)
    selected=dict(arm=arm,tier=tier,suite_sha256=new_suite['sha256'],full_replication=bool(tier),
                  patch_sha256=read(root/arm/'build.json')['patch_sha256'])
    if (dest/'selected.json').exists():
        assert read(dest/'selected.json')==selected
    write(dest/'selected.json',selected)
    results={'parent':{},arm:{}}
    with serve(root,arm,dest) as (plan,health):
        for i,group in enumerate(order(new_suite) if tier else FOCUS):
            for model in (('parent',arm) if i%2==0 else (arm,'parent')):
                note(root,'replication',model,group)
                results[model].update(run_group(new_suite,group,model,plan,dest))
    if tier:
        full_report(dest,new_suite,results['parent'],results[arm],arm)
    write(dest/'focus.json',compare_groups(results['parent'],results[arm],FOCUS))
    note(root,'complete',arm,'all')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--study', choices=('edits', 'step-pairs'), default='edits')
    args=parser.parse_args()
    evaluate(args.root, ARMS if args.study=='edits' else ('step-pairs',))
