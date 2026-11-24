"""Chunked upload of a large file to a box through the Jupyter contents API. usage: BOX=N upload_big.py LOCAL REMOTE_PATH"""
import sys, os, base64, importlib.util
spec = importlib.util.spec_from_file_location('gpu', os.path.join(os.path.dirname(__file__), 'gpu.py')); g = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv; sys.argv = ['gpu.py', 'noop']; spec.loader.exec_module(g); sys.argv = sys.argv_saved
local, remote = sys.argv[1], sys.argv[2]
CH = int(os.environ.get("CHUNK_KB", "640")) * 1024; size = os.path.getsize(local); n = (size + CH - 1) // CH
with open(local, 'rb') as f:
    for i in range(1, n + 1):
        data = base64.b64encode(f.read(CH)).decode()
        r = g.client.put(f'{g.BASE}/api/contents/{remote}', json={'type': 'file', 'format': 'base64', 'content': data, 'chunk': -1 if i == n else i})
        if r.status_code >= 300: print('fail', i, r.status_code, r.text[:200]); sys.exit(1)
        print(i, '/', n, flush=True) if i % 50 == 0 or i == n else None
print('uploaded', size)
