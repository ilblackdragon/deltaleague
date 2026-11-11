import copy,json
import pytest
from delta_cli.public_eval.suite import PROTOCOL,manifest,expected_items

def public_result(content_hash):
    return {'protocol':PROTOCOL,'suite_sha256':manifest()['sha256'],'content_hash':content_hash,
            'reference_check':False,'groups':{name:[{'id':i,'score':1.0} for i in sorted(ids)]
                for name,ids in expected_items(manifest()).items()}}

def test_public_runner_replays_only_bfcl_and_keeps_other_evidence(tmp_path,monkeypatch):
    from delta_cli.public_eval import runner,tools
    source={'public_validation':public_result('a'*64),'public_evidence':{'gsm8k':{'preserved':True}}}
    monkeypatch.setattr(runner,'check_versions',lambda:None)
    monkeypatch.setattr(runner,'version',lambda name:'fixture')
    calls=[]
    def bfcl(suite,endpoint,model,tokenizer,output,**kwargs):
        calls.append('bfcl');directory=output.with_suffix('');directory.mkdir()
        for category,spec in suite['bfcl'].items():
            entries=[{'correct_count':0}]+[{'id':i} for i in spec['ids']]
            (directory/f'BFCL_v4_{category}_score.json').write_text('\n'.join(map(json.dumps,entries)))
            (directory/f'{category}.responses.json').write_text('[]')
        return {}
    monkeypatch.setattr(tools,'run',bfcl)
    result=runner.evaluate_public(endpoint='http://fixture',model='candidate',tokenizer='fixture',content_hash='a'*64,
        output=tmp_path/'run',reuse_non_bfcl=source)
    assert calls==['bfcl']
    groups=result['public_validation']['groups']
    for k,rows in groups.items():
        if not k.startswith('bfcl'):assert rows==source['public_validation']['groups'][k]
        else:assert all(r['score']==0 for r in rows)
    assert result['public_evidence']['gsm8k']==source['public_evidence']['gsm8k']
    bad=copy.deepcopy(source);bad['public_validation']['content_hash']='b'*64
    with pytest.raises(ValueError):runner.evaluate_public(endpoint='http://fixture',model='candidate',tokenizer='fixture',content_hash='a'*64,output=tmp_path/'bad',reuse_non_bfcl=bad)
