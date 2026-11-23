"""Build verified attention-scale and seed-diversified checkpoint edits."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from delta_cli.artifacts import build_sparse, file_hash, reconstruct, verify_artifact_bytes

LEADER='917d6b4f-fea3-432c-9264-e4f89e45890a'
DONORS={'top-eff1e1b7/model':'eff1e1b7-18ef-4c77-83ef-25fdcc027e75',
        'models/a6eb9f13':'a6eb9f13-47fb-4755-bd08-268a86307674',
        'models/fdfd46a4':'fdfd46a4-3a55-4277-91c1-70c739d6139b'}
TARGETS={f'model.language_model.layers.{n}.self_attn.q_norm.weight' for n in (11,15,19)}
ARMS=('query-sharp','query-soft','seed-three')


def verified(directory,version):
    info=json.loads((directory/'delta-model.json').read_text())
    assert info['id']==version
    verify_artifact_bytes(directory,info['files'])
    return info


def build(root,arm):
    exp=root.parent
    parent=exp/'models/917d6b4f'
    meta=verified(parent,LEADER)
    assert meta['content_hash']=='2c94041a49dfc1fbf74c1214487dd10dd72e288f5a5e0b6d8fff6d432bb9b59f'
    out=root/arm
    out.mkdir()
    model=out/'model'
    model.mkdir()
    torch.set_num_threads(2)
    filename='model.safetensors'
    changed={}
    effective={}
    with ExitStack() as stack:
        base=stack.enter_context(safe_open(parent/filename,framework='pt'))
        donors=[]
        if arm=='seed-three':
            for path,version in DONORS.items():
                verified(exp/path,version)
                donors.append(stack.enter_context(safe_open(exp/path/filename,framework='pt')))
        tensors={}
        for name in base.keys():
            original=base.get_tensor(name)
            value=original
            if arm!='seed-three' and name in TARGETS:
                factor=1.08 if arm=='query-sharp' else 0.92
                # Stored Qwen RMSNorm is an offset from unit gain.
                gain=1+original.float()
                value=(factor*gain-1).to(original.dtype)
                effective[name]=float(torch.linalg.vector_norm(1+value.float())/torch.linalg.vector_norm(gain))
            elif arm=='seed-three':
                match=re.match(r'model\.language_model\.layers\.(\d+)\.',name)
                if match and int(match[1])<23:
                    eff,a6,seed=(d.get_tensor(name).float() for d in donors)
                    assert torch.equal(((eff+a6)*0.5).to(original.dtype),original)
                    value=(eff*0.5+a6*0.25+seed*0.25).to(original.dtype)
            if value is not original:
                changed_count=int((value!=original).sum())
                if changed_count:
                    changed[name]=changed_count
            tensors[name]=value
        assert changed and (arm=='seed-three' or set(changed)==TARGETS)
        save_file(tensors,model/filename,metadata=base.metadata())
    for f in meta['files']:
        if f['name']!=filename:
            shutil.copyfile(parent/f['name'],model/f['name'])
    submission=out/'submission'
    manifest=build_sparse(parent/filename,model/filename,submission,LEADER,meta['content_hash'],meta['model_id'])
    assert set(manifest['tensors'])==set(changed)
    patch=submission/'weights.safetensors'
    rebuilt=reconstruct(parent/filename,patch,manifest,allowed_prefix='model.language_model.')
    with safe_open(model/filename,framework='pt') as saved:
        assert set(saved.keys())==set(rebuilt)
        for key,weight in rebuilt.items():
            assert torch.equal(saved.get_tensor(key),weight)
    report=dict(model=str(model),submission=str(submission),official_content_hash=None,parent_content_hash=meta['content_hash'],
                checkpoint_sha256=file_hash(model/filename),patch_sha256=file_hash(patch),manifest_sha256=file_hash(submission/'manifest.json'),
                patch_bytes=patch.stat().st_size,changed_values=sum(changed.values()))
    (model/'local-candidate.json').write_text(json.dumps(report,indent=2))
    plan=json.loads((exp/'science-dpo-v1/plan.json').read_text())
    plan.update(candidates={arm:report},purpose='Joint reasoning and general transfer',promotion_rule='See protocol.json',
                medium_confidence_rule='See protocol.json')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'build.json').write_text(json.dumps(report|dict(changed_tensors=len(changed),effective_q_norm_ratio=effective,
                official_reconstruction_exact=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('arm',choices=ARMS)
    a=p.parse_args()
    build(a.root,a.arm)
