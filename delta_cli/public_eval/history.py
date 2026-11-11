"""Content-addressed response history, shared by subset and full-set evaluators."""
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def namespace(files, runtime, server):
    return digest({'version': 1, 'files': sorted((f['name'], f['sha256']) for f in files),
                   'runtime': runtime, 'server': server})


class History:
    def __init__(self, path, *, seed_path=None):
        self.seed_path = Path(seed_path) if seed_path else None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS responses (namespace TEXT, key TEXT, value TEXT, PRIMARY KEY(namespace,key))')
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, scope, key):
        with self.connect() as db:
            row = db.execute('SELECT value FROM responses WHERE namespace=? AND key=?', (scope,key)).fetchone()
        if row:
            return json.loads(row[0])
        if self.seed_path:
            # A shared immutable checkpoint is read once per needed request. Keep
            # only used responses in each task's own evidence/cache export.
            from urllib.parse import quote
            db = sqlite3.connect('file:'+quote(str(self.seed_path.resolve()))+'?mode=ro&immutable=1', uri=True)
            try:
                row = db.execute('SELECT value FROM responses WHERE namespace=? AND key=?', ('public',key)).fetchone()
            finally:
                db.close()
            if row:
                value = json.loads(row[0])
                if not valid_response(value):raise ValueError('Invalid seeded response')
                self.put(scope,key,value)
                return value
        return None

    def put(self, scope, key, value):
        raw = json.dumps(value, allow_nan=False)
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO responses VALUES (?,?,?)', (scope,key,raw))

    def import_rows(self, scope, rows):
        with self.connect() as db:
            db.executemany('INSERT OR IGNORE INTO responses VALUES (?,?,?)',
                           ((scope,key,json.dumps(value,allow_nan=False)) for key,value in rows.items()))

    def rows(self, scope):
        with self.connect() as db:
            return {key: json.loads(value) for key,value in db.execute('SELECT key,value FROM responses WHERE namespace=?', (scope,))}


def valid_response(value):
    if not isinstance(value, dict):
        return False
    choices = value.get('choices')
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            return False
        reason = choice.get('finish_reason')
        if reason not in ('stop', 'length', 'content_filter', 'tool_calls'):
            return False
        if isinstance(choice.get('text'), str):
            continue
        message = choice.get('message')
        if not isinstance(message, dict):
            return False
        if isinstance(message.get('content'), str):
            continue
        if reason in ('length', 'content_filter') and message.get('content') is None:
            continue
        if reason == 'tool_calls' and isinstance(message.get('tool_calls'), list) and message['tool_calls']:
            continue
        return False
    return True


class ResponseProxy:
    """Loopback OpenAI proxy: only complete successful responses enter history.

    Keys contain the entire wire request, including logprob, tool and seed options.
    The caller supplies a checkpoint/runtime namespace. No benchmark IDs or scores.
    """
    def __init__(self, history, scope, seed=None, upstream='http://127.0.0.1:8000', *, request_timeout=1800, observer=None, upstream_model=None, api_key_env=None):
        self.history, self.scope, self.upstream = history, scope, upstream
        import os
        api_key = os.environ[api_key_env] if api_key_env else None
        headers = {'Authorization': 'Bearer '+api_key} if api_key else {}
        self.hits = self.misses = 0
        self.lock = threading.Lock()
        self.locks = [threading.Lock() for _ in range(128)]
        history.import_rows(scope, {key:value for key,value in (seed or {}).items() if valid_response(value)})
        import httpx
        self.client=httpx.Client(timeout=request_timeout,limits=httpx.Limits(max_connections=128,max_keepalive_connections=32))
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version="HTTP/1.1"
            disable_nagle_algorithm=True
            def log_message(self, *args):
                pass

            def do_POST(self):
                import httpx
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 < size <= 32 * 1024**2:
                        raise ValueError('Invalid request size')
                    request = json.loads(self.rfile.read(size))
                    if self.path not in ('/v1/completions','/v1/chat/completions') or request.get('stream'):
                        raise ValueError('Unsupported inference request')
                    if upstream_model:
                        request['model'] = upstream_model
                    key = digest({'path': self.path, 'request': request})
                    with owner.locks[int(key[:8],16) % len(owner.locks)]:
                        response = history.get(scope, key)
                        if response is None:
                            result = owner.client.post(owner.upstream+self.path, json=request, timeout=request_timeout, headers=headers)
                            if result.status_code != 200:
                                self.send_response(result.status_code);self.send_header('Content-Length',str(len(result.content)));self.end_headers();self.wfile.write(result.content)
                                return
                            response = result.json()
                            if not valid_response(response):
                                raise ValueError('Incomplete inference response')
                            history.put(scope, key, response)
                            with owner.lock: owner.misses += 1
                        else:
                            with owner.lock: owner.hits += 1
                    if observer is not None:
                        observer(request, response)
                    body = json.dumps(response).encode()
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                except Exception:
                    self.send_response(502);self.send_header('Content-Length','0');self.send_header('Connection','close');self.end_headers();self.close_connection=True
        class Server(ThreadingHTTPServer):
            request_queue_size=128
        self.server = Server(('127.0.0.1',0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    @property
    def endpoint(self):
        return 'http://127.0.0.1:'+str(self.server.server_port)+'/v1'

    def __exit__(self, *args):
        self.server.shutdown();self.server.server_close();self.thread.join();self.client.close()


def export_checkpoint(history, scope, destination):
    """Stream a single public namespace into a compressed, consistent SQLite snapshot."""
    import gzip
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        target = History(Path(td)/'snapshot.sqlite3')
        with history.connect() as source, target.connect() as out:
            out.executemany('INSERT INTO responses VALUES (?,?,?)',
                (('public', key, value) for key,value in source.execute(
                    'SELECT key,value FROM responses WHERE namespace=?', (scope,))))
        temp = Path(str(destination)+'.tmp')
        with target.path.open('rb') as source, gzip.open(temp, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(source, out)
        temp.replace(destination)


def import_checkpoint(history, scope, source):
    """Validate and stream a scoped checkpoint without loading responses into RAM."""
    import gzip
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'snapshot.sqlite3'
        with gzip.open(source, 'rb') as stream, path.open('wb') as dest:
            shutil.copyfileobj(stream, dest)
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            with history.connect() as target:
                for namespace, key, raw in db.execute('SELECT namespace,key,value FROM responses'):
                    if namespace != 'public' or len(key) != 64 or any(c not in '0123456789abcdef' for c in key) or not valid_response(json.loads(raw)):
                        raise ValueError('Invalid public checkpoint')
                    target.execute('INSERT OR IGNORE INTO responses VALUES (?,?,?)', (scope,key,raw))
        finally:
            db.close()
