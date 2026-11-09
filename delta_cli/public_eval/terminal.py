"""Standalone Terminal-Bench 2.1 runner, native Harbor trials and durable checkpoints."""
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BENCHMARK = 'terminal_bench_2_1'


def context_tokens(tokenizer, messages):
    # Harbor counts an empty Chat before populating its initial messages.
    # There is no prompt to render yet; HF chat templates reject an empty list.
    if not messages:return 0
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=True,enable_thinking=True)
    # Transformers 5 returns BatchEncoding for this tokenizer. len(encoded)
    # counts fields (usually two), not tokens; older versions return a list.
    tokens=encoded['input_ids'] if isinstance(encoded,Mapping) else encoded
    if tokens and isinstance(tokens[0],list):
        if len(tokens)!=1:raise ValueError('Expected one conversation for token counting')
        tokens=tokens[0]
    return len(tokens)


def trial_name(root, task, attempt, retry):
    # Harbor uses the trial name as its Docker Compose project. Different
    # concurrent runs of the same case must never share a sandbox name.
    run_key=hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
    return f'{task}__a{attempt}r{retry}__{run_key}'


def atomic_json(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,default=str));temp.replace(path)


def normalized_request(request, protocol, attempt):
    request=dict(request)
    for key in ('max_completion_tokens','reasoning_effort','stream_options'):
        request.pop(key,None)
    request.update(model='candidate',stream=False,max_tokens=8192,
        seed=protocol['seed']+attempt,chat_template_kwargs={'enable_thinking':True})
    for key in ('temperature','top_p','top_k','presence_penalty'):
        request[key]=protocol['generation'][key]
    return request


def final_response(response):
    # Preserve raw thinking in the scoped inference cache; agents consume final text.
    response=json.loads(json.dumps(response))
    for choice in response['choices']:
        message=choice['message'];text=message.get('content') or ''
        message['content']=text.rsplit('</think>',1)[1].strip() if '</think>' in text else ''
        message.pop('reasoning_content',None)
    return response


@contextmanager
def agent_endpoint(proxy, protocol, attempt):
    import httpx
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length',0))
                if self.path!='/v1/chat/completions' or not 0<size<=32*1024**2:
                    raise ValueError('Invalid inference request')
                request=normalized_request(json.loads(self.rfile.read(size)),protocol,attempt)
                r=httpx.post(proxy.endpoint+'/chat/completions',json=request,timeout=None)
                if r.status_code!=200:
                    self.send_response(r.status_code);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(r.content);return
                body=json.dumps(final_response(r.json())).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception:
                self.send_response(502);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:server.shutdown();server.server_close();thread.join()


def trial_config(task_path, trials_dir, trial_name, endpoint, protocol, *, agent='terminus-2'):
    from harbor.models.trial.config import TrialConfig,TaskConfig,AgentConfig,EnvironmentConfig
    kwargs={**protocol['agent_options'],'temperature':protocol['generation']['temperature'],
        'model_info':{'max_input_tokens':protocol['max_model_len']-8192,'max_output_tokens':8192,
            'input_cost_per_token':0,'output_cost_per_token':0,'litellm_provider':'openai','mode':'chat'},
        'llm_kwargs':{'api_key':'local-inference'},'llm_call_kwargs':{'max_tokens':8192}}
    return TrialConfig(task=TaskConfig(path=task_path),trials_dir=trials_dir,trial_name=trial_name,
        agent=AgentConfig(name=agent,model_name='openai/candidate' if agent=='terminus-2' else None,
            base_url=endpoint if agent=='terminus-2' else None,
            model_api_format='openai_chat_completions' if agent=='terminus-2' else None,
            kwargs=kwargs if agent=='terminus-2' else {}),environment=EnvironmentConfig(type='docker',delete=True))


def valid_reward(result):
    """Infrastructure errors never enter the score denominator as model failures."""
    if not result.reward_valid:
        raise RuntimeError('Invalid native reward: '+str(result.failure_cause))
    reward=result.verifier_result.rewards.get('reward')
    if type(reward) not in (int,float) or reward not in (0,1):
        raise ValueError('Terminal-Bench requires a binary native reward')
    return float(reward)



def validate_verifier_evidence(evidence):
    """Pinned Terminal-Bench tests emit CTRF even when assertions fail.

    A shell reward of zero without a test report is a bootstrap failure, not
    evidence of a failed model solution. Collection errors remain native zeros.
    """
    try:
        report=json.loads(evidence['verifier/ctrf.json'])['results']
        summary=report['summary'];tests=report['tests']
        if not isinstance(tests,list) or not isinstance(summary,dict):raise ValueError()
        for field in ('tests','passed','failed'):
            if type(summary.get(field)) is not int or summary[field]<0:raise ValueError()
    except (KeyError,TypeError,ValueError) as exc:
        raise RuntimeError('Verifier infrastructure failure: missing or malformed native CTRF report') from exc
    return report


async def prepare_verifier_dependencies(trial, evidence_path):
    """Refresh Debian metadata and wait for package locks before native tests.

    Runs after the agent, changes no task artifacts or native assertions, and
    records its output separately. The native verifier still installs its own
    exact dependencies and computes its own reward.
    """
    command="""set -eu
if command -v apt-get >/dev/null; then
  printf 'DPkg::Lock::Timeout "600";\\nAcquire::Retries "3";\\n' > /etc/apt/apt.conf.d/99-terminal-verifier-retries
  # Bullseye security package versions remain available in Debian's archive.
  if grep -q 'VERSION_CODENAME=bullseye' /etc/os-release; then
    for source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
      [ ! -f "$source" ] || sed -i 's|http://deb.debian.org/debian-security|https://archive.debian.org/debian-security|g; s|https://deb.debian.org/debian-security|https://archive.debian.org/debian-security|g; s|http://security.debian.org/debian-security|https://archive.debian.org/debian-security|g; s|https://security.debian.org/debian-security|https://archive.debian.org/debian-security|g' "$source"
    done
  fi
  apt-get -o APT::Update::Error-Mode=any update
fi
"""
    result=await trial.agent_environment.exec(command,timeout_sec=900,user='root')
    evidence_path.parent.mkdir(parents=True,exist_ok=True)
    evidence_path.write_text(str(result.stdout or '')+'\n'+str(result.stderr or ''))
    if result.return_code:raise RuntimeError('Verifier dependency preparation failed')

def load_checkpoint(path, protocol_hash, protocol):
    if not path.exists():return {}
    with gzip.open(path,'rt') as f:data=json.load(f)
    if data['protocol_hash']!=protocol_hash:raise ValueError('Terminal checkpoint identity mismatch')
    completed=data['completed']
    if not isinstance(completed,dict):raise ValueError('Invalid terminal checkpoint')
    allowed={f'{task}/{attempt}' for task in protocol['task_ids'] for attempt in range(protocol['attempts_per_task'])}
    if set(completed)-allowed:raise ValueError('Unexpected terminal checkpoint tasks')
    for key,row in completed.items():
        if type(row.get('reward')) not in (int,float) or row['reward'] not in (0,1):
            raise ValueError('Invalid checkpoint reward')
        if key!=f"{row['task']}/{row['attempt']}":raise ValueError('Checkpoint task mismatch')
    return completed


def task_evidence(directory):
    # Native result, agent trajectories and verifier logs, including failed retries.
    files={}
    for path in directory.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.txt','.log'):
            if path.stat().st_size<=32*1024**2:
                files[str(path.relative_to(directory))]=path.read_text(errors='replace')
    return files


def prepare_dataset(root, protocol):
    import io,tarfile,httpx
    manifest=json.loads((root/'terminal-manifest.json').read_text())
    if manifest['revision']!=protocol['dataset_revision']:raise ValueError('Dataset revision mismatch')
    dataset=root/'dataset'
    if not dataset.exists():
        config=json.loads((root/'manifest.json').read_text())
        mirror=config.get('terminal_dataset_source')
        if mirror and mirror['revision']!=manifest['revision']:raise ValueError('Terminal mirror revision mismatch')
        url=mirror['url'] if mirror else 'https://codeload.github.com/harbor-framework/terminal-bench-2-1/tar.gz/'+manifest['revision']
        response=httpx.get(url,timeout=180)
        response.raise_for_status()
        if mirror and hashlib.sha256(response.content).hexdigest()!=mirror['sha256']:raise ValueError('Terminal mirror integrity')
        with tarfile.open(fileobj=io.BytesIO(response.content),mode='r:gz') as archive:
            archive.extractall(root/'source',filter='data')
        (root/'source'/('terminal-bench-2-1-'+manifest['revision'])/'tasks').rename(dataset)
    for task in protocol['task_ids']:
        expected=manifest['tasks'][task]['files']
        actual={str(f.relative_to(dataset/task)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (dataset/task).rglob('*') if f.is_file()}
        if actual!=expected:raise ValueError('Task file integrity: '+task)
    return dataset


def evaluate(root):
    from .history import History,ResponseProxy,export_checkpoint,import_checkpoint
    from harbor.trial.trial import Trial
    from .native_trial import pin_simple_docker_images
    pin_simple_docker_images()
    config=json.loads((root/'manifest.json').read_text());p=dict(config['protocol'])
    selected=config.get('terminal_keys')
    if selected:
        p['task_ids']=sorted({k.rsplit('/',1)[0] for k in selected});p['expected_items']=len(selected)
    if p.get('context_counting')=='qwen-tokenizer-v1':
        from transformers import AutoTokenizer
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        tokenizer=AutoTokenizer.from_pretrained(str(root/'model'),trust_remote_code=False)
        def exact_count(self,chat):
            return context_tokens(tokenizer,chat.messages)
        Terminus2._count_total_tokens=exact_count
    dataset=prepare_dataset(root,p)
    task_manifest=json.loads((root/'terminal-manifest.json').read_text())
    history=History(root/'history.sqlite3',seed_path=config.get('history_seed_path'))
    with history.connect() as db:db.execute('PRAGMA journal_mode=WAL')
    if not config.get('history_seed_path'):import_checkpoint(history,'public',root/'history-seed.sqlite3.gz')
    completed=load_checkpoint(root/'terminal-seed.json.gz',config['protocol_hash'],p)
    report=root/(BENCHMARK+'.report.json.gz')
    counts=dict(requests=0,prompt_tokens=0,completion_tokens=0,truncated=0)
    started=time.monotonic();lock=threading.RLock();failures={}
    raw=gzip.open(root/(BENCHMARK+'.raw.jsonl.gz'),'wt')
    def progress():
        atomic_json(root/'progress.json',{'stage':'evaluating','benchmark':BENCHMARK,
            'benchmark_completed':len(completed),'benchmark_total':p['expected_items'],
            'requests_completed':counts['requests'],'tokens':counts['completion_tokens'],
            'benchmark_seconds':time.monotonic()-started,'infrastructure_retries':sum(len(v) for v in failures.values())})
    def checkpoint():
        temp=report.with_name(report.name+'.tmp')
        with gzip.open(temp,'wt') as f:json.dump({'protocol_hash':config['protocol_hash'],
            'completed':completed,'failed_attempts':failures,'counts':counts},f,default=str)
        temp.replace(report)
    def observe(request,response):
        with lock:
            raw.write(json.dumps({'request':request,'response':response})+'\n')
            counts['requests']+=1
            for key in ('prompt_tokens','completion_tokens'):counts[key]+=response.get('usage',{}).get(key,0)
            counts['truncated']+=sum(c.get('finish_reason')=='length' for c in response['choices'])
            progress()
    stop=threading.Event()
    def snapshot():
        while not stop.wait(300):
            try:export_checkpoint(history,'public',root/'history.sqlite3.gz')
            except Exception:traceback.print_exc()
    thread=threading.Thread(target=snapshot,daemon=True);thread.start()
    try:
        with ResponseProxy(history,'public',upstream=config.get('inference_upstream','http://127.0.0.1:8000'), upstream_model=config.get('inference_model'), api_key_env=config.get('api_key_env'),request_timeout=None,observer=observe) as proxy:
            async def run_all():
                semaphore=asyncio.Semaphore(p['task_concurrency'])
                async def one(task,attempt):
                    key=f'{task}/{attempt}'
                    if key in completed or (selected and key not in selected):return
                    async with semaphore:
                        with agent_endpoint(proxy,p,attempt) as endpoint:
                            for retry in range(4):
                                name=trial_name(root,task,attempt,retry)
                                cfg=trial_config(dataset/task,root/'trials',name,endpoint,p)
                                try:
                                    trial=await Trial.create(cfg)
                                    from harbor.trial.hooks import TrialEvent
                                    async def verifier_ready(event):
                                        await prepare_verifier_dependencies(trial,root/'trials'/name/'verifier-bootstrap.txt')
                                    trial.add_hook(TrialEvent.VERIFICATION_START,verifier_ready)
                                    result=await trial.run()
                                    evidence=task_evidence(root/'trials'/name)
                                    validate_verifier_evidence(evidence)
                                    reward=valid_reward(result)
                                    native=result.model_dump(mode='json')
                                    image=task_manifest['tasks'][task]['environment'].get('docker_image')
                                    inspection=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True) if image else None
                                    image_identity=json.loads(inspection.stdout) if inspection and inspection.returncode==0 else []
                                    row={'task':task,'attempt':attempt,'reward':reward,'native_result':native,'container_image_identity':image_identity,
                                        'evidence':evidence}
                                    with lock:completed[key]=row;checkpoint();progress()
                                    if image:
                                        try:subprocess.run(['docker','image','rm',image],capture_output=True,timeout=120)
                                        except Exception:traceback.print_exc()
                                    return
                                except Exception as exc:
                                    with lock:
                                        failures.setdefault(key,[]).append({'error_type':type(exc).__name__,
                                            'error':str(exc),'evidence':task_evidence(root/'trials'/name)})
                                        checkpoint();progress()
                                    if retry==3:raise RuntimeError('Infrastructure retries exhausted: '+key) from exc
                # Finish healthy trials even if another task exhausts its retries.
                results=await asyncio.gather(*(one(t,a) for t in p['task_ids'] for a in range(p['attempts_per_task'])),return_exceptions=True)
                if any(isinstance(r,BaseException) for r in results):
                    raise RuntimeError('Some terminal tasks have no valid native reward; resume checkpoint')
            progress();checkpoint();asyncio.run(run_all())
            if len(completed)!=p['expected_items']:raise ValueError('Incomplete terminal coverage')
            summary={BENCHMARK:{'completed_items':len(completed),'expected_items':p['expected_items'],
                'metrics':{'primary_metric':'mean pass@1','score_percent':100*sum(r['reward'] for r in completed.values())/len(completed),
                    'attempts_per_task':p['attempts_per_task'],'tasks':{k:{'reward':r['reward']} for k,r in completed.items()}},
                'performance':{**counts,'wallclock_seconds':time.monotonic()-started,'reused_requests':proxy.hits,
                    'new_requests':proxy.misses,'restored_trials':len(load_checkpoint(root/'terminal-seed.json.gz',config['protocol_hash'],p))}}}
            atomic_json(root/'summary.json',summary)
    finally:
        stop.set();thread.join();raw.close()
        export_checkpoint(history,'public',root/'history.sqlite3.gz')
        with lock:checkpoint()

