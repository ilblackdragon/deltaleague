"""Transfer the frozen two-parent attention edit to the official three-parent mixture."""
import json
import sys
from pathlib import Path

import torch
from delta_cli.artifacts import file_hash, verify_artifact_bytes

from build import MODEL_FILE, build_candidate, verified, write


def build_transfer(root, protocol):
    study = Path(__file__).resolve().parent
    rule = json.loads(protocol.read_text())
    source_report = json.loads((study / 'results/builds.json').read_text())['attention-toward-grpo']
    source_parent_report = json.loads((study / 'results/parent-identity.json').read_text())
    models = root.parent / 'models'
    leader = models / rule['parent'][:8]
    leader_meta = verified(leader, rule['content_hash'])
    assert leader_meta['id'] == rule['parent']
    old_parent = models / rule['transfer_parent'][:8]
    old_meta = verified(old_parent, source_parent_report['content_hash'])
    assert old_meta['id'] == rule['transfer_parent']
    source = root.parent / 'leader96-mixture-v1/attention-toward-grpo'
    assert file_hash(source / 'model' / MODEL_FILE) == rule['transfer_source_sha256']
    assert source_report['patch_sha256'] == rule['transfer_patch_sha256'] == file_hash(source / 'submission/weights.safetensors')
    template = json.loads((source / 'plan.json').read_text())
    template['generation']['seed'] = rule['development']['seed']
    arm, spec = next(iter(rule['arms'].items()))
    torch.set_num_threads(2)
    report = build_candidate(root, arm, spec, leader, leader_meta, (old_parent, source / 'model'), template)
    verify_artifact_bytes(leader, leader_meta['files'])
    assert 0 < report['changed_values'] <= source_report['changed_values']
    source_tensors = source_report['by_tensor']
    eligible = sum(source_tensors[name]['changed'] for name in report['by_tensor'])
    assert sum(row['source_differences'] for row in report['by_tensor'].values()) == eligible
    write(root / 'builds.json', {arm: report})
    print(arm, report['changed_values'], report['checkpoint_sha256'], flush=True)


if __name__ == '__main__':
    protocol = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / 'transfer/protocol.json'
    build_transfer(Path(sys.argv[1]).resolve(), protocol)
