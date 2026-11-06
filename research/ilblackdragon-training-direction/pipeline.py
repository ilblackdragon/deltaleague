"""Sequential, bounded orchestration. No submission, rental, or parameter search."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
w=p.parse_args().workspace.resolve();source=Path(__file__).resolve().parent
plan=json.loads((w/'plan.json').read_text())
lock=(w/'local/pipeline.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
def record(status,**extra):
    (w/'local/pipeline-status.json').write_text(json.dumps({'status':status,'updated_at_unix':time.time(),**extra},indent=2)+'\n')
def stage(name):
    remaining=plan['deadline_unix']-time.time()
    if remaining<=0:raise TimeoutError('Approved wall-clock ceiling exhausted')
    record('running',stage=name)
    with (w/f'{name}.log').open('a') as log:
        subprocess.run([sys.executable,str(source/'run.py'),'--workspace',str(w),'--stage',name],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=True)
def artifact_check(command):
    remaining=plan['deadline_unix']-time.time()
    if remaining<=0:raise TimeoutError('Approved wall-clock ceiling exhausted')
    with (w/'artifact-checks.log').open('a') as log:
        subprocess.run(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=True)
try:
    for name in ['train_student','train_teacher','edit']:stage(name)
    if not (w/'edited.safetensors').exists():
        record('no_supported_edit')
    else:
        delta=str(Path(sys.executable).with_name('delta'))
        artifact_check([delta,'patch','build','--parent','./model','--modified','./edited.safetensors','--output','./submission'])
        artifact_check([delta,'validate','./submission'])
        artifact_check([sys.executable,str(source/'audit.py'),'--workspace',str(w)])
        # No holdout observations until edit is immutable. Preserve all three models.
        for name in ['heldout_student','heldout_candidate','heldout_teacher']:stage(name)
        record('completed_local_comparison')
except (TimeoutError,subprocess.TimeoutExpired) as e:
    record('budget_exhausted',detail=str(e))
except subprocess.CalledProcessError as e:
    record('budget_exhausted' if time.time()>=plan['deadline_unix']-2 else 'stage_failed',returncode=e.returncode,detail=str(e))
    raise
finally:
    subprocess.run([sys.executable,str(source/'report.py'),'--workspace',str(w)],check=True)
