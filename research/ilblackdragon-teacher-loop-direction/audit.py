"""Validate the parent and the exact, separately saved direct-edit checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from delta_cli.artifacts import file_hash, reconstruct


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();w=args.workspace.resolve();torch.set_num_threads(16)
    metadata=json.loads((w/'model/delta-model.json').read_text())
    for f in metadata['files']:
        path=w/'model'/f['name']
        assert path.stat().st_size==f['size_bytes'] and file_hash(path)==f['sha256']
    if not (w/'edited.safetensors').exists():
        report={'parent_files_verified':len(metadata['files']),'candidate_exists':False}
    else:
        before=load_file(w/'model/model.safetensors');after=load_file(w/'edited.safetensors')
        manifest=json.loads((w/'submission/manifest.json').read_text())
        assert manifest['parent_version_id']==metadata['id'] and manifest['parent_content_hash']==metadata['content_hash']
        rebuilt=reconstruct(w/'model/model.safetensors',w/'submission/weights.safetensors',manifest)
        assert before.keys()==after.keys()==rebuilt.keys()
        changed=[]
        for key,t in before.items():
            assert t.shape==after[key].shape and t.dtype==after[key].dtype
            assert torch.isfinite(after[key]).all() and torch.equal(after[key],rebuilt[key])
            if not torch.equal(t,after[key]):
                changed.append(key)
        assert set(changed)=={f'model.language_model.layers.{i}.mlp.down_proj.weight' for i in [15,19]}
        report={'parent_files_verified':len(metadata['files']),'candidate_exists':True,
            'changed_tensors':changed,'reconstruction_bitwise_equal':True,
            'edited_sha256':file_hash(w/'edited.safetensors'),
            'patch_sha256':file_hash(w/'submission/weights.safetensors'),
            'patch_bytes':(w/'submission/weights.safetensors').stat().st_size}
    (w/'local/audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
