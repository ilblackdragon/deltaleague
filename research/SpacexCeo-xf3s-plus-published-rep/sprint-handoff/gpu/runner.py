"""Serial GPU job runner: picks queue/*.json (oldest first), builds the candidate with apply_edit.py
(or uses spec['model'] directly), runs eval_vllm.py, writes results/<name>.json, moves spec to done/."""
import json, os, glob, subprocess, time, shutil, sys
H = os.path.expanduser('~/dl'); os.chdir(H)
for d in ['queue', 'done', 'results', 'cands', 'failed']: os.makedirs(d, exist_ok=True)
PY = os.path.expanduser('~/venv/bin/python')
while True:
    jobs = sorted(glob.glob('queue/*.json'), key=os.path.getmtime)
    if not jobs:
        if os.path.exists('STOP'): break
        time.sleep(10); continue
    j = jobs[0]; name = os.path.basename(j)[:-5]
    open("RUNNING", "w").write(name)
    spec = json.load(open(j)); t0 = time.time()
    log = open(f'results/{name}.log', 'w')
    try:
        model = spec.get('model')
        if not model:
            spec.setdefault('base', 'versions/0ea62f40-7cd8-4d98-9ebb-484192fa05eb/model')
            spec['out'] = f'cands/{name}'
            json.dump(spec, open(f'results/{name}.spec.json', 'w'))
            subprocess.run([PY, 'apply_edit.py', f'results/{name}.spec.json'], stdout=log, stderr=subprocess.STDOUT, check=True)
            model = spec['out']
        subprocess.run([PY, 'eval_vllm.py', model, spec.get('prompts', 'proxy400.json'), f'results/{name}.json',
                        str(spec.get('samples', 1)), str(spec.get('max_tokens', 8192))], stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.move(j, f'done/{name}.json')
        if spec.get('cleanup', True) and not spec.get('model'): shutil.rmtree(f'cands/{name}', ignore_errors=True)
    except Exception as e:
        log.write(f'\nFAILED {e}\n'); shutil.move(j, f'failed/{name}.json')
    log.write(f'\nelapsed {time.time()-t0:.0f}s\n'); log.close(); open('RUNNING', 'w').write('')
