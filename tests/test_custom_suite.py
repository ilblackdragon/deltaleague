from types import SimpleNamespace
import pytest
from delta_cli.public_eval import custom_suite


def test_coding_uses_shared_request_and_sandbox(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr('httpx.post',lambda url,**kw:(calls.append((url,kw)) or SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'</think>{"files":{"solution.py":"def solve(x): return x"}}'},'finish_reason':'stop'}],'usage':{'completion_tokens':11}})))
    monkeypatch.setattr(custom_suite,'check_python',lambda files,inputs,expected,**kw: {'passed':inputs==expected,'reason':'passed'})
    row,evidence=custom_suite.evaluate_item({'id':'coding','track':'coding','prompt':'echo','starter_files':{'solution.py':''},'hidden_cases':[{'input':3,'expected':3}]},endpoint='http://fixture/private/v1/chat/completions',generation={'max_tokens':8192,'enable_thinking':True,'temperature':1,'top_p':.95,'top_k':20,'presence_penalty':1.5},seed=42,root=tmp_path,tokenizer=tmp_path,harbor_root=tmp_path,image='fixture')
    assert row['passed'] and row['completion_tokens']==11
    assert calls[0][1]['json']['seed']==42 and calls[0][1]['timeout'] is None
    assert evidence['response']['choices']


def test_infrastructure_failure_propagates(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('sandbox unavailable')
    monkeypatch.setattr(custom_suite,'evaluate_item',fail)
    with pytest.raises(RuntimeError,match='unavailable'):custom_suite.evaluate([{'id':'one'}])
