#!/usr/bin/env python3
"""Run shell commands on the rented GPU box through its Jupyter REST/websocket API.
Usage: gpu.py sh "<cmd>" [timeout_s] | gpu.py put <local> <remote> | gpu.py get <remote> <local>
Credentials are read from $GPU_AUTH_FILE (user:password), never printed."""
import sys, os, json, uuid, base64, time, ssl, httpx, websocket
SCR = '/tmp/claude-0/-home-user/35252158-b758-51e1-8ca0-55b144f16305/scratchpad'
BOX = os.environ.get('BOX', '1')
TOKEN = None
cfg = json.load(open(os.environ.get('GPU_BOX_DIR', SCR) + f'/.box{BOX}.json'))  # {"host","base", "token" | "basic_file"}
if 'basic_file' in cfg:
    AUTH = open(cfg['basic_file']).read().strip()
    HOST = cfg['host']; PATH = cfg['base']
    user, pw = AUTH.split(':', 1)
    client = httpx.Client(auth=(user, pw), timeout=120, verify=os.environ.get('SSL_CERT_FILE', '/root/.ccr/ca-bundle.crt'))
else:
    HOST = cfg['host']; PATH = cfg['base']; TOKEN = cfg['token']; AUTH = None
    client = httpx.Client(timeout=120, verify='/root/.ccr/ca-bundle.crt', headers={'Authorization': f'token {TOKEN}'})
BASE = f'https://{HOST}{PATH}'
KFILE = SCR + f'/.gpu_kernel{BOX}'
client.get(f'{BASE}/lab' if AUTH else f'{BASE}/api/status')
XSRF = client.cookies.get('_xsrf', '')
client.headers['X-XSRFToken'] = XSRF
COOKIE = '; '.join(f'{k}={v}' for k, v in client.cookies.items())

def kernel():
    if os.path.exists(KFILE):
        kid = open(KFILE).read().strip()
        if client.get(f'{BASE}/api/kernels/{kid}').status_code == 200:
            return kid
    kid = client.post(f'{BASE}/api/kernels', json={'name': 'python3'}).json()['id']
    open(KFILE, 'w').write(kid)
    return kid

def execute(code, timeout=600):
    kid = kernel()
    hdr = ('Basic ' + base64.b64encode(AUTH.encode()).decode()) if AUTH else f'token {TOKEN}'
    proxy = os.environ.get('HTTPS_PROXY', '').replace('http://', '')
    ph, pp = proxy.split(':') if proxy else (None, None)
    ws = websocket.create_connection(f'wss://{HOST}{PATH}/api/kernels/{kid}/channels', header=[f'Authorization: {hdr}', f'Cookie: {COOKIE}'],
                                     http_proxy_host=ph, http_proxy_port=pp, proxy_type='http',
                                     sslopt={'ca_certs': '/root/.ccr/ca-bundle.crt'}, timeout=timeout)
    mid = uuid.uuid4().hex
    ws.send(json.dumps({'header': {'msg_id': mid, 'username': 'x', 'session': uuid.uuid4().hex, 'msg_type': 'execute_request', 'version': '5.3'},
                        'parent_header': {}, 'metadata': {}, 'channel': 'shell',
                        'content': {'code': code, 'silent': False, 'store_history': False, 'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True}}))
    end = time.time() + timeout
    while time.time() < end:
        m = json.loads(ws.recv())
        if m.get('parent_header', {}).get('msg_id') != mid: continue
        t, c = m['msg_type'], m['content']
        if t == 'stream': sys.stdout.write(c['text']); sys.stdout.flush()
        elif t in ('execute_result', 'display_data'): print(c['data'].get('text/plain', ''))
        elif t == 'error': print('\n'.join(c['traceback'])); 
        elif t == 'status' and c['execution_state'] == 'idle': break
    ws.close()

def sh(cmd, timeout=600):
    code = ('import subprocess,sys\n'
            f'p=subprocess.run(["bash","-lc",{cmd!r}],capture_output=True,text=True,timeout={timeout})\n'
            'print(p.stdout[-20000:]);print(p.stderr[-6000:],file=sys.stderr);print("[exit]",p.returncode)')
    execute(code, timeout + 30)

if __name__ == '__main__':
    op = sys.argv[1]
    if op == 'restart':
        kid = kernel(); print(client.post(f'{BASE}/api/kernels/{kid}/restart').status_code)
    elif op == 'bg':
        code = ('import subprocess\n'
                f'subprocess.Popen(["bash","-lc",{sys.argv[2]!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n'
                'print("started")')
        execute(code, 60)
    elif op == 'sh': sh(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    elif op == 'py': execute(open(sys.argv[2]).read(), int(sys.argv[3]) if len(sys.argv) > 3 else 600)
    elif op == 'put':
        data = base64.b64encode(open(sys.argv[2], 'rb').read()).decode()
        r = client.put(f'{BASE}/api/contents/{sys.argv[3]}', json={'type': 'file', 'format': 'base64', 'content': data}); print(r.status_code)
    elif op == 'get':
        r = client.get(f'{BASE}/api/contents/{sys.argv[2]}', params={'content': 1}).json()
        c = r['content']; open(sys.argv[3], 'wb').write(base64.b64decode(c) if r['format'] == 'base64' else c.encode()); print('ok')
