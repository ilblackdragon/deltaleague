import pytest
from delta_cli.public_eval.custom_tasks import candidate_files,chat_request,answer

def test_coding_prompt_uses_real_paths():
    from delta_cli.public_eval.custom_tasks import coding_prompt
    prompt=coding_prompt('Implement solve.',{'solution.py':'pass','helpers.py':'pass'})
    assert 'relative/path.py' not in prompt
    assert '"files": {"solution.py":' in prompt and '"helpers.py":' in prompt

def test_malformed_json_is_not_python_fallback():
    with pytest.raises(ValueError):candidate_files('{"files":{"solution.py":"def solve(x):', ['solution.py'])
    with pytest.raises(ValueError):candidate_files('', ['solution.py'])
    with pytest.raises(ValueError):candidate_files('explanation then code', ['solution.py'])

def test_file_allowlist_and_traversal():
    with pytest.raises(ValueError):candidate_files('{"files":{"../solution.py":"pass"}}',['../solution.py'])
    with pytest.raises(ValueError):candidate_files('{"files":{"secret.py":"pass"}}',['solution.py'])
    assert candidate_files('```python\ndef solve(x): return x\n```',['solution.py'])['solution.py'].startswith('def solve')

def test_thinking_and_decoding():
    config={'max_tokens':8192,'enable_thinking':True,'temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':1.5}
    req=chat_request('test',config,42)
    assert req['seed']==42 and req['max_tokens']==8192 and req['top_k']==20
    assert answer({'choices':[{'message':{'content':'reasoning</think> 7'}}]})=='7'
    assert answer({'choices':[{'message':{'content':'unfinished reasoning'}}]})==''

def test_agent_budget_includes_every_call(monkeypatch):
    import json,urllib.request,urllib.error
    import httpx
    from delta_cli.public_eval.custom_tasks import bounded_agent_endpoint
    calls=[]
    def upstream(url,**kwargs):
        body=kwargs['json'];calls.append(body)
        tokens=8100 if len(calls)==1 else 92
        return httpx.Response(200,json={'choices':[{'message':{'content':'reason</think>done'}}], 'usage':{'completion_tokens':tokens,'prompt_tokens':10}})
    monkeypatch.setattr(httpx,'post',upstream)
    p={'seed':10,'generation':{'temperature':1,'top_p':.95,'top_k':20,'presence_penalty':1.5}}
    with bounded_agent_endpoint('http://unused',p,0,output_limit=8192) as (endpoint,stats):
        for _ in range(2):
            request=urllib.request.Request(endpoint+'/chat/completions',data=json.dumps({'messages':[]}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:assert json.load(response)['choices'][0]['message']['content']=='done'
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==400 and stats['budget_exhausted']
    assert [c['max_tokens'] for c in calls]==[8192,92]
