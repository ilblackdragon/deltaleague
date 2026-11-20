#!/usr/bin/env python3
"""Client for the serial GPU queue on the rented RTX 5090 (see AGENT_BRIEF.md).
  q.py submit NAME SPEC.json   queue a job (NAME: [a-z0-9_-], prefix with your agent tag, e.g. a2_tokset_v1)
  q.py status                  queue, running job, finished/failed jobs
  q.py result NAME [DIR]       download results/NAME.json (+ .texts.jsonl if --texts) into DIR (default ./results)
  q.py log NAME                tail of the job log"""
import sys, os, json, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__)); GPU = [sys.executable, os.path.join(HERE, 'gpu.py')]
def sh(cmd, t=120): return subprocess.run(GPU + ['sh', cmd, str(t)], capture_output=True, text=True).stdout
op = sys.argv[1]
if op == 'submit':
    name, spec = sys.argv[2], sys.argv[3]
    assert re.fullmatch(r'[a-z0-9_-]{3,60}', name), 'bad name'
    json.load(open(spec))
    tmp = f'dl/incoming_{name}.json'
    subprocess.run(GPU + ['put', spec, tmp], check=True, capture_output=True)
    print(sh(f'cd $HOME/dl; if [ -e done/{name}.json ] || [ -e queue/{name}.json ] || [ -e failed/{name}.json ]; then echo NAME_TAKEN; rm incoming_{name}.json; else mv incoming_{name}.json queue/{name}.json && echo QUEUED; fi'))
elif op == 'status':
    print(sh('cd $HOME/dl; echo "QUEUE:"; ls -tr queue; echo "RUNNING:"; cat RUNNING 2>/dev/null; echo; echo "DONE:"; ls -t done | head -40; echo "FAILED:"; ls failed'))
elif op == 'result':
    name = sys.argv[2]; d = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 'results'
    os.makedirs(d, exist_ok=True)
    subprocess.run(GPU + ['get', f'dl/results/{name}.json', f'{d}/{name}.json'], check=True)
    if '--texts' in sys.argv: subprocess.run(GPU + ['get', f'dl/results/{name}.texts.jsonl', f'{d}/{name}.texts.jsonl'], check=True)
    print(json.dumps(json.load(open(f'{d}/{name}.json'))['summary'], indent=1))
elif op == 'log':
    print(sh(f'cd $HOME/dl; grep -v Warning results/{sys.argv[2]}.log | tail -c 3000'))
