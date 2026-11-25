"""Reproduce the submitted sparse patch directly from four official checkpoints."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

from delta_cli.artifacts import build_sparse, file_hash
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from build import MODEL_FILE, SOURCE_HASHES, edit_tensor, verified

HERE = Path(__file__).resolve().parent


def reproduce(models, output):
    if output.exists():
        raise FileExistsError(f'Choose a new output path: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rule = json.loads((HERE / 'ablation/protocol.json').read_text())
    reference = json.loads((HERE / 'ablation/results/builds.json').read_text())['full-attention-transfer']
    midpoint = json.loads((HERE / 'results/parent-identity.json').read_text())
    content = {**SOURCE_HASHES, '96b67023': midpoint['content_hash'], 'd0120ae6': rule['content_hash']}
    for key, digest in content.items():
        verified(models / key, digest)
    spec = dict(next(iter(rule['arms'].values())), family='mixture', donor_fraction=0.625)
    parent_path = models / 'd0120ae6' / MODEL_FILE
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp, ExitStack() as stack:
        weights = {key: stack.enter_context(safe_open(models / key / MODEL_FILE, framework='pt')) for key in content}
        parent = weights['d0120ae6']
        tensors = {name: parent.get_tensor(name) for name in parent.keys()}
        for layer in (11, 15, 19):
            name = f'model.language_model.layers.{layer}.self_attn.o_proj.weight'
            old = weights['96b67023'].get_tensor(name)
            blend, _ = edit_tensor(name, old, spec, (weights['917d6b4f'], weights['a6eb9f13']))
            tensors[name] = (tensors[name].float() + (blend.float() - old.float())).to(old.dtype)
        modified = Path(tmp) / MODEL_FILE
        save_file(tensors, modified, metadata=parent.metadata())
        artifact = Path(tmp) / 'submission'
        build_sparse(parent_path, modified, artifact, rule['parent'], rule['content_hash'], 'qwen35-08b-v1')
        assert file_hash(artifact / 'weights.safetensors') == reference['patch_sha256']
        assert file_hash(artifact / 'manifest.json') == reference['manifest_sha256']
        artifact.rename(output)
    print(reference['patch_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    reproduce(options.models, options.output)
