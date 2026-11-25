"""Reconstruct and byte-verify edits made directly to the official mixture checkpoint."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from delta_cli.artifacts import build_sparse, file_hash, reconstruct, verify_artifact_bytes
from safetensors import safe_open
from safetensors.torch import save_file

RULE = json.loads(Path(__file__).with_name('protocol.json').read_text())
MODEL_FILE = 'model.safetensors'
PREFIX = 'model.language_model.layers.'
SOURCE_HASHES = {
    '917d6b4f':'2c94041a49dfc1fbf74c1214487dd10dd72e288f5a5e0b6d8fff6d432bb9b59f',
    'a6eb9f13':'2d1c20e8f75a77b78c45918f6db646f9730d64d47768fa3af913988f09f97bcf',
}


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def verified(directory, content):
    meta = json.loads((directory/'delta-model.json').read_text())
    assert meta['id'].startswith(directory.name) and meta['content_hash'] == content
    assert meta['model_id'] == 'qwen35-08b-v1'
    verify_artifact_bytes(directory, meta['files'])
    return meta


def edit_tensor(name, original, spec, sources):
    if not name.startswith(PREFIX):
        return None
    layer_text, module = name.removeprefix(PREFIX).split('.',1)
    layer = int(layer_text)
    if spec['family'] == 'nonrotary-query':
        if layer not in spec['layers'] or module != 'self_attn.q_norm.weight':
            return None
        assert original.shape == (256,) and original.dtype == torch.bfloat16
        value = original.clone()
        value[64:] = ((1+original[64:].float())*spec['factor']-1).to(original.dtype)
        return value, {'eligible':192}
    if not spec['first_layer'] <= layer <= spec['last_layer'] or module not in spec['modules']:
        return None
    before, after = (source.get_tensor(name) for source in sources)
    assert before.shape == after.shape == original.shape and before.dtype == after.dtype == original.dtype
    differs = before != after
    a, b = before.float(), after.float()
    if spec['family'] == 'transfer-delta':
        value = torch.where(differs, (original.float() + (b - a)).to(original.dtype), original)
        return value, {'source_differences': int(differs.sum()), 'eligible': int(differs.sum())}
    eligible = differs & (original == ((a+b)*0.5).to(original.dtype))
    f = spec['donor_fraction']
    blended = ((1-f)*a+f*b).to(original.dtype)
    value = torch.where(eligible, blended, original)
    return value, {'source_differences':int(differs.sum()),'eligible':int(eligible.sum())}


def build_candidate(root, arm, spec, leader, meta, source_dirs, template):
    dest = root/arm
    if dest.exists():
        saved = json.loads((dest/'build.json').read_text())
        assert saved['parent_content_hash']==meta['content_hash'] and saved['edit']==spec
        for key, path in [('checkpoint_sha256',dest/'model'/MODEL_FILE),('patch_sha256',dest/'submission/weights.safetensors'),
                          ('manifest_sha256', dest/'submission/manifest.json')]:
            assert saved[key] == file_hash(path)
        return saved
    with tempfile.TemporaryDirectory(dir=root) as temp:
        stage=Path(temp)/arm
        model=stage/'model'
        model.mkdir(parents=True)
        audit={}
        with safe_open(leader/MODEL_FILE,framework='pt') as parent, \
             safe_open(source_dirs[0]/MODEL_FILE,framework='pt') as left, \
             safe_open(source_dirs[1]/MODEL_FILE,framework='pt') as right:
            tensors={}
            for name in parent.keys():
                original=parent.get_tensor(name)
                edited=edit_tensor(name,original,spec,(left,right))
                if edited is None:
                    tensors[name]=original
                else:
                    value,row=edited
                    assert value.shape==original.shape and value.dtype==original.dtype and torch.isfinite(value).all()
                    row['changed']=int((value!=original).sum())
                    tensors[name]=value
                    audit[name]=row
            assert sum(row['changed'] for row in audit.values())>0
            save_file(tensors,model/MODEL_FILE,metadata=parent.metadata())
            del tensors
        for item in meta['files']:
            if item['name']!=MODEL_FILE:
                shutil.copyfile(leader/item['name'],model/item['name'])
        submission=stage/'submission'
        manifest=build_sparse(leader/MODEL_FILE,model/MODEL_FILE,submission,meta['id'],meta['content_hash'],meta['model_id'])
        assert set(manifest['tensors'])=={k for k,v in audit.items() if v['changed']}
        patch=submission/'weights.safetensors'
        rebuilt=reconstruct(leader/MODEL_FILE,patch,manifest,allowed_prefix='model.language_model.')
        with safe_open(model/MODEL_FILE,framework='pt') as saved:
            assert set(saved.keys())==set(rebuilt)
            assert all(torch.equal(saved.get_tensor(k),v) for k,v in rebuilt.items())
        identity=dict(model=str(dest/'model'),submission=str(dest/'submission'),official_content_hash=None,
                      parent_content_hash=meta['content_hash'],checkpoint_sha256=file_hash(model/MODEL_FILE),
                      patch_sha256=file_hash(patch),manifest_sha256=file_hash(submission/'manifest.json'),
                      patch_bytes=patch.stat().st_size,changed_values=sum(v['changed'] for v in audit.values()))
        write(model/'local-candidate.json',identity)
        write(stage/'plan.json',dict(template,parent=str(leader),parent_content_hash=meta['content_hash'],
              candidates={arm:identity},purpose='Transfer and local blend ratios on the official leaderboard parent'))
        report=dict(identity,edit=spec,by_tensor=audit,official_reconstruction_exact=True)
        write(stage/'build.json',report)
        stage.rename(dest)
    return report


def main(root):
    torch.set_num_threads(2)
    models=root.parent/'models'
    leader=models/RULE['parent'][:8]
    meta=verified(leader,RULE['content_hash'])
    assert meta['id']==RULE['parent']
    config=json.loads((leader/'config.json').read_text())['text_config']
    assert config['head_dim']==256 and int(config['head_dim']*config['rope_parameters']['partial_rotary_factor'])==64
    sources=[models/name for name in RULE['sources']]
    for directory in sources:
        verified(directory,SOURCE_HASHES[directory.name])
    template=json.loads((root.parent/'reasoning-balance-v1/query-sharp/plan.json').read_text())
    reports={}
    for arm,spec in RULE['arms'].items():
        report=build_candidate(root,arm,spec,leader,meta,sources,template)
        reports[arm]=report
        print(arm,report['changed_values'],report['patch_sha256'],flush=True)
    assert len({r['checkpoint_sha256'] for r in reports.values()})==len(reports)
    verify_artifact_bytes(leader,meta['files'])
    write(root/'builds.json',reports)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
