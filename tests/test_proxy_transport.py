import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import httpx
from delta_cli.public_eval.history import History,ResponseProxy


def test_reuses_upstream_connection_and_preserves_uncached_errors(tmp_path):
    ports=[];calls=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1';disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            ports.append(self.client_address[1]);calls.append(request)
            failed=len(calls)==1
            body=json.dumps({'error':'temporary'} if failed else {'choices':[{'text':'ok','finish_reason':'stop'}]}).encode()
            self.send_response(503 if failed else 200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with ResponseProxy(History(tmp_path/'cache'),'public',upstream=f'http://127.0.0.1:{server.server_port}') as proxy,httpx.Client(timeout=5) as client:
            url=proxy.endpoint+'/completions';request={'prompt':'one','model':'candidate'}
            assert client.post(url,json=request).status_code==503
            assert client.post(url,json=request).status_code==200
            assert client.post(url,json=request).status_code==200
            assert client.post(url,json={**request,'prompt':'two'}).status_code==200
            assert proxy.hits==1 and proxy.misses==2
        assert len(calls)==3 and len(set(ports))==1
    finally:server.shutdown();server.server_close();thread.join()
