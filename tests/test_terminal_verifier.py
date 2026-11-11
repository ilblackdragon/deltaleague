import asyncio
import json
from types import SimpleNamespace
import pytest
from delta_cli.public_eval.terminal import validate_verifier_evidence,prepare_verifier_dependencies


def test_bootstrap_zero_is_not_native_failure():
    for evidence in ({'verifier/reward.txt':'0'}, {'verifier/ctrf.json':'invalid'}, {'verifier/ctrf.json':'{}'}):
        with pytest.raises(RuntimeError,match='Verifier infrastructure'):validate_verifier_evidence(evidence)


def test_assertion_and_collection_failures_remain_valid():
    for n in [0,3]:
        data={'results':{'summary':{'tests':n,'passed':0,'failed':n},'tests':[]}}
        assert validate_verifier_evidence({'verifier/ctrf.json':json.dumps(data)})==data['results']


def test_dependency_failure_is_explicit_and_logged(tmp_path):
    calls=[]
    async def execute(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(stdout='apt failed',stderr='network error',return_code=100)
    trial=SimpleNamespace(agent_environment=SimpleNamespace(exec=execute))
    with pytest.raises(RuntimeError,match='dependency preparation'):
        asyncio.run(prepare_verifier_dependencies(trial,tmp_path/'bootstrap.txt'))
    assert 'network error' in (tmp_path/'bootstrap.txt').read_text()
    assert 'DPkg::Lock::Timeout "600"' in calls[0][0]
