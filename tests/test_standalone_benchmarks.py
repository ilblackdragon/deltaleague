"""Public execution contract: no controller, secrets or fleet required."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from delta_cli.public_eval import standalone
from delta_cli.public_eval.coverage import expand, slice_suite
from delta_cli.public_eval.suite import manifest

GENERATION = {'enable_thinking': True, 'max_tokens': 8192, 'temperature': .6,
              'top_p': .95, 'top_k': 20, 'presence_penalty': 0.0}


def test_full_and_sample_native_populations_and_shards():
    suite = manifest()
    assert standalone.make_protocol('ifeval', GENERATION, coverage='full')['expected_items'] == 541
    assert standalone.make_protocol('ifeval', GENERATION)['expected_items'] == suite['text']['ifeval']['count']
    full = expand(suite, 'humaneval')
    assert len(full['code']['humaneval']['docs']) == 164
    result = []
    for start in range(0, 164, 31):
        shard = slice_suite(full, 'humaneval', [start, min(start+31,164)])
        result.extend(shard['code']['humaneval']['docs'])
    assert result == full['code']['humaneval']['docs']
    assert full['code']['humaneval']['count'] == 164
    for bounds in ([-1,2],[1,1],[1,165],[0,True]):
        with pytest.raises(ValueError):slice_suite(full, 'humaneval', bounds)


def test_terminal_protocol_is_public_and_selects_native_tasks():
    p = standalone.make_protocol(standalone.TERMINAL, GENERATION, coverage='full', attempts=5)
    assert p['expected_items'] == 89*5
    sample = standalone.make_protocol(standalone.TERMINAL, GENERATION, task_ids=p['task_ids'][:2])
    assert sample['expected_items'] == 2
    with pytest.raises(ValueError):standalone.make_protocol(standalone.TERMINAL, GENERATION)
    with pytest.raises(ValueError):standalone.make_protocol('not-a-benchmark', GENERATION)


def test_standalone_has_no_controller_dependency(tmp_path, monkeypatch):
    tokenizer=tmp_path/'tokenizer';tokenizer.mkdir()
    output=tmp_path/'out';monkeypatch.setenv('TEST_MODEL_KEY','do-not-store-me')
    protocol=standalone.make_protocol('ifeval',GENERATION,coverage='full')
    def execute(root):
        config=json.loads((root/'manifest.json').read_text())
        assert config['inference_upstream']=='https://model.invalid'
        assert config['inference_model']=='external-model'
        assert config['task_count']==2 and config['task_slice']==[3,5]
        assert config['api_key_env']=='TEST_MODEL_KEY'
        assert (root/'history-seed.sqlite3.gz').exists()
        assert (root/'model').resolve()==tokenizer
        return {'ifeval':{'completed_items':2}}
    monkeypatch.setattr(standalone,'execute',execute)
    result=standalone.run_benchmark(protocol=protocol,endpoint='https://model.invalid/v1',model='external-model',
        tokenizer=tokenizer,content_hash='a'*64,output=output,bounds=[3,5],api_key_env='TEST_MODEL_KEY')
    assert result['ifeval']['completed_items']==2
    assert 'do-not-store-me' not in (output/'manifest.json').read_text()
    assert 'content_hash' not in protocol
    with pytest.raises(FileExistsError):
        standalone.run_benchmark(protocol=protocol,endpoint='https://model.invalid/v1',model='external-model',
            tokenizer=tokenizer,content_hash='a'*64,output=output)


def test_cli_protocol_and_shard_contract(tmp_path):
    from delta_cli.public_eval.cli import app
    settings=tmp_path/'generation.json';settings.write_text(json.dumps(GENERATION))
    runner=CliRunner()
    result=runner.invoke(app,['benchmark-config','--benchmark','ifeval','--generation',str(settings),'--coverage','full'])
    assert result.exit_code==0,result.output
    assert json.loads(result.stdout)['expected_items']==541
    result=runner.invoke(app,['benchmark','--help'])
    from click import unstyle
    assert result.exit_code==0 and '--api-key-env' in unstyle(result.stdout)


def test_external_transport_preserves_error_and_remaps_model(tmp_path, monkeypatch):
    import httpx
    from delta_cli.public_eval.history import History,ResponseProxy
    original=httpx.Client.post;calls=[]
    monkeypatch.setenv('TEST_MODEL_KEY','private-token')
    def post(self,url,**kwargs):
        if url.startswith('http://127.0.0.1:'):return original(self,url,**kwargs)
        calls.append(kwargs)
        return httpx.Response(400,json={'error':{'code':'context_length_exceeded'}})
    monkeypatch.setattr(httpx.Client,'post',post)
    with ResponseProxy(History(tmp_path/'cache.sqlite3'),'public',upstream='https://model.invalid',
                       upstream_model='external-model',api_key_env='TEST_MODEL_KEY') as proxy:
        response=httpx.post(proxy.endpoint+'/chat/completions',json={'model':'candidate','messages':[]})
    assert response.status_code==400
    assert calls[0]['json']['model']=='external-model'
    assert calls[0]['headers']=={'Authorization':'Bearer private-token'}
    assert response.json()['error']['code']=='context_length_exceeded'


def test_shared_seed_exports_only_responses_used_by_the_task(tmp_path):
    from delta_cli.public_eval.history import History
    seed=History(tmp_path/'seed.sqlite3')
    response={'choices':[{'text':'done','finish_reason':'stop'}]}
    seed.put('public','needed',response);seed.put('public','unrelated',response)
    task=History(tmp_path/'task.sqlite3',seed_path=seed.path)
    assert task.get('public','needed')==response
    assert task.rows('public')=={'needed':response}
    assert set(seed.rows('public'))=={'needed','unrelated'}
    assert task.get('public','missing') is None


def test_terminal_context_count_handles_harbor_initial_empty_chat():
    from delta_cli.public_eval.terminal import context_tokens
    class Tokenizer:
        def apply_chat_template(self,messages,**options):
            assert messages==[{'role':'user','content':'hello'}]
            assert options=={'tokenize':True,'add_generation_prompt':True,'enable_thinking':True}
            return [11,12,13]
    tokenizer=Tokenizer()
    assert context_tokens(tokenizer,[])==0
    assert context_tokens(tokenizer,[{'role':'user','content':'hello'}])==3


@pytest.mark.parametrize('encoded',[
    {'input_ids':[11,12,13,14],'attention_mask':[1,1,1,1]},
    {'input_ids':[[11,12,13,14]],'attention_mask':[[1,1,1,1]]},
])
def test_terminal_context_counts_tokens_not_transformers_encoding_fields(encoded):
    from delta_cli.public_eval.terminal import context_tokens
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return encoded
    assert context_tokens(Tokenizer(),[{'role':'user','content':'hello'}])==4


def test_terminal_sandbox_names_isolate_concurrent_runs_and_retries(tmp_path):
    from delta_cli.public_eval.terminal import trial_name
    first=trial_name(tmp_path/'one','task',0,0)
    assert first==trial_name(tmp_path/'one','task',0,0)
    assert first!=trial_name(tmp_path/'two','task',0,0)
    assert first!=trial_name(tmp_path/'one','task',0,1)
