"""Check parent integrity, exact patch reconstruction, and one-tensor edit scope."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from delta_cli.artifacts import file_hash, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    torch.set_num_threads(16)
    metadata = json.loads((w/'model/delta-model.json').read_text())
    for f in metadata['files']:
        path = w/'model'/f['name']
        assert path.stat().st_size == f['size_bytes']
        assert file_hash(path) == f['sha256']
    before = load_file(w/'model/model.safetensors')
    after = load_file(w/'edited.safetensors')
    manifest = json.loads((w/'submission/manifest.json').read_text())
    rebuilt = reconstruct(w/'model/model.safetensors', w/'submission/weights.safetensors', manifest)
    assert before.keys() == after.keys() == rebuilt.keys()
    changed = []
    for key, tensor in before.items():
        assert tensor.dtype == after[key].dtype and tensor.shape == after[key].shape
        assert torch.isfinite(after[key]).all()
        assert torch.equal(after[key], rebuilt[key]), key
        if not torch.equal(tensor, after[key]):
            changed.append(key)
    assert changed == ['model.language_model.layers.23.mlp.down_proj.weight']
    target = changed[0]
    tensor_hash = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    report = {'parent_files_verified': len(metadata['files']), 'tensor_count': len(before),
              'changed_tensors': changed, 'reconstruction_bitwise_equal': True,
              'parent_target_sha256': tensor_hash(before[target]),
              'edited_target_sha256': tensor_hash(after[target]),
              'patch_sha256': file_hash(w/'submission/weights.safetensors'),
              'patch_bytes': (w/'submission/weights.safetensors').stat().st_size,
              'manifest_sha256': file_hash(w/'submission/manifest.json')}
    (w/'research-local/audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
