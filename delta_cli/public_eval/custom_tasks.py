"""Generic custom-task primitives. No competition data or scheduling policy.

Candidate programs run with only candidate files and test inputs. Expected
outputs stay in the trusted caller, outside the execution container.
"""
import json
import ast
import keyword
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
from .generation import final_answer, sampling


def chat_request(prompt, generation, seed):
    return {'model':'candidate','messages':[{'role':'user','content':prompt}],
            'max_tokens':generation['max_tokens'],'seed':seed,'stream':False,
            'chat_template_kwargs':{'enable_thinking':generation['enable_thinking']},
            **sampling(generation)}


def answer(response, thinking=True):
    message=response['choices'][0]['message'];text=message.get('content') or ''
    return final_answer(text) if thinking else text.strip()


def coding_prompt(prompt, starter_files):
    """Use real filenames in the output example; placeholders become model output."""
    if not starter_files:raise ValueError('No candidate files')
    example={'files':{name:'complete Python source for this file' for name in starter_files}}
    return (prompt+'\nExisting files:\n'+json.dumps(starter_files)+
            '\nReturn only a JSON object with this structure: '+json.dumps(example)+
            '\nUse exactly the filenames shown above. Include complete contents of changed files. '+
            'Encode newlines as JSON escapes inside strings. Do not include explanations.')


def python_entrypoint(entrypoint=None):
    """Validate the trusted call contract; preserve legacy solve(input) tasks."""
    spec={'module':'solution','function':'solve','call_style':'single'}
    if entrypoint is not None:
        if not isinstance(entrypoint,dict) or set(entrypoint)-set(spec):
            raise ValueError('Invalid Python entrypoint')
        spec.update(entrypoint)
    for field in ('module','function'):
        value=spec[field]
        if not isinstance(value,str) or not value.isascii() or not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError('Invalid Python entrypoint identifier')
    if spec['call_style'] not in ('single','args_kwargs'):
        raise ValueError('Invalid Python call style')
    return spec


def python_bootstrap(entrypoint=None):
    spec=python_entrypoint(entrypoint)
    call='fn(x)' if spec['call_style']=='single' else "fn(*x.get('args',[]),**x.get('kwargs',{}))"
    return ("import sys,json,importlib;sys.path.insert(0,'/candidate');"
            f"fn=getattr(importlib.import_module({spec['module']!r}),{spec['function']!r});"
            f"print(json.dumps([{call} for x in json.load(sys.stdin)],allow_nan=False))")


def candidate_files(text, allowed, *, entrypoint=None):
    spec=python_entrypoint(entrypoint)
    match=re.fullmatch(r'\s*```(?:json|python)?\s*\n(.*?)\n```\s*',text,re.S)
    text=match.group(1) if match else text
    try:
        data=json.loads(text)
        files=data.get('files') if isinstance(data,dict) else None
    except ValueError:
        # Only accept actual Python as the single-file fallback. Malformed JSON
        # used to be executed as Python and mislabeled as an execution failure.
        files=None
        if set(allowed)=={spec['module']+'.py'}:
            try:tree=ast.parse(text)
            except SyntaxError:pass
            else:
                if any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==spec['function'] for node in tree.body):
                    files={spec['module']+'.py':text}
    if not isinstance(files,dict) or not files or set(files)-set(allowed):
        raise ValueError('Invalid candidate file set')
    for name,source in files.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(source,str) or len(source.encode())>256_000:
            raise ValueError('Invalid candidate file')
    return files


def check_python(files, inputs, expected, *, image, timeout=30, entrypoint=None):
    if len(inputs)!=len(expected):raise ValueError('Case count mismatch')
    spec=python_entrypoint(entrypoint)
    if spec['module']+'.py' not in files:raise ValueError('Missing Python entrypoint file')
    if spec['call_style']=='args_kwargs':
        for case in inputs:
            if (not isinstance(case,dict) or set(case)-{'args','kwargs'}
                or not isinstance(case.get('args',[]),list)
                or not isinstance(case.get('kwargs',{}),dict)
                or any(not isinstance(k,str) for k in case.get('kwargs',{}))):
                raise ValueError('Invalid Python call arguments')
    name='delta-custom-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='delta-candidate-') as td:
        root=Path(td);root.chmod(0o755)
        for name_in,source in files.items():
            p=Path(name_in)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe candidate path')
            target=root/p;target.parent.mkdir(parents=True,exist_ok=True)
            for directory in target.parents:
                if directory==root:break
                directory.chmod(0o755)
            target.write_text(source);target.chmod(0o444)
        # No hidden reference data is mounted or embedded in this bootstrap.
        bootstrap=python_bootstrap(spec)
        command=['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
                 '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','32',
                 '--memory','512m','--memory-swap','512m','--cpus','1','--user','65534:65534',
                 '--ulimit','cpu=10:10','--ulimit','fsize=1048576:1048576',
                 '--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
                 '--mount',f'type=bind,src={root},dst=/candidate,readonly',
                 '--entrypoint','python3',image,'-I','-B','-c',bootstrap]
        try:
            with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
                process=subprocess.Popen(['prlimit','--fsize=2097152:2097152','--',*command],stdin=subprocess.PIPE,stdout=stdout,stderr=stderr)
                try:process.communicate(json.dumps(inputs).encode(),timeout=timeout)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','kill',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
                    process.kill();process.communicate();return {'passed':False,'reason':'timeout'}
                if process.returncode in (125,126,127):raise RuntimeError('Candidate sandbox unavailable')
                stdout.seek(0);raw=stdout.read(2*1024**2)
                if process.returncode:return {'passed':False,'reason':'execution_failure'}
                try:actual=json.loads(raw)
                except (ValueError,UnicodeError):return {'passed':False,'reason':'invalid_output'}
                # JSON encoding distinguishes true/1 and rejects non-finite values.
                try:passed=json.dumps(actual,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False)
                except ValueError:passed=False
                return {'passed':passed,'reason':'passed' if passed else 'wrong_answer'}
        finally:
            subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)


from contextlib import contextmanager
@contextmanager
def bounded_agent_endpoint(upstream, protocol, attempt, *, output_limit=65536, call_limit=128, budget_stop_path=None):
    """Bound a multi-turn agent including summarization calls; expose usage."""
    import threading
    import httpx
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from .terminal import normalized_request,final_response
    stats={'calls':0,'prompt_tokens':0,'completion_tokens':0,'budget_exhausted':False,'inference_seconds':0.0}
    import time
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length',0))
                if self.path!='/v1/chat/completions' or not 0<size<=2*1024**2:raise ValueError('request')
                remaining=output_limit-stats['completion_tokens']
                if remaining<=0 or stats['calls']>=call_limit:
                    stats['budget_exhausted']=True
                    if budget_stop_path is not None:Path(budget_stop_path).write_text('agent_budget_exhausted')
                    self.send_response(400);self.end_headers();self.wfile.write(b'{"error":{"message":"agent budget exhausted","type":"budget_exhausted"}}');return
                request=normalized_request(json.loads(self.rfile.read(size)),protocol,attempt)
                request['max_tokens']=min(request['max_tokens'],remaining);stats['calls']+=1
                start=time.monotonic()
                response=httpx.post(upstream,json=request,timeout=None)
                stats['inference_seconds']+=time.monotonic()-start
                if response.status_code!=200:
                    self.send_response(response.status_code);self.end_headers();self.wfile.write(response.content);return
                data=response.json()
                for key in ('prompt_tokens','completion_tokens'):stats[key]+=data.get('usage',{}).get(key,0)
                body=json.dumps(final_response(data)).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception:
                self.send_response(502);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',stats
    finally:server.shutdown();server.server_close();thread.join()
