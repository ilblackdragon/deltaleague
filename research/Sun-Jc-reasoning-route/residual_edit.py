"""Compare individual seed updates against the fixed three-source leader."""
from delta_cli.artifacts import verify_artifact_bytes
import json
import torch


def seed_sources(root, rule):
    ancestor = root.parent / 'top-eff1e1b7/model'
    meta = json.loads((ancestor / 'delta-model.json').read_text())
    assert meta['id'] == rule['ancestor'] and meta['content_hash'] == rule['ancestor_hash']
    verify_artifact_bytes(ancestor, meta['files'])
    return ancestor, root.parent / 'models' / rule['donor'][:8]


def residual_tensor(name, leader, spec, sources, *, agreement=None):
    if spec['family'] == 'final-channel-transfer':
        return final_channel_tensor(name, leader, spec, sources)
    prefix = 'model.language_model.layers.'
    if not name.startswith(prefix) or leader.ndim != 2:
        return None
    number, module = name.removeprefix(prefix).split('.', 1)
    if not spec['first_layer'] <= int(number) <= spec['last_layer']:
        return None
    if not (module.endswith('_proj.weight') or module.startswith('linear_attn.in_proj_') and module.endswith('.weight')):
        return None
    if 'modules' in spec and module not in spec['modules']:
        return None
    ancestor, donor = (file.get_tensor(name) for file in sources)
    assert leader.dtype == ancestor.dtype == donor.dtype and leader.shape == ancestor.shape == donor.shape
    family = spec['family']
    if family == 'consensus-floor':
        reference = agreement.get_tensor(name)
        assert reference.dtype == leader.dtype and reference.shape == leader.shape
        lower = torch.minimum(reference, donor)
        upper = torch.maximum(reference, donor)
        move_up = (ancestor < lower) & (leader < lower)
        move_down = (ancestor > upper) & (leader > upper)
        changed = torch.where(move_up, lower, torch.where(move_down, upper, leader))
        return changed, {'eligible': int((move_up | move_down).sum())}
    parent = leader.float()
    old = ancestor.float()
    seed_delta = donor.float() - old
    if family == 'missing-seed':
        eligible = (leader == ancestor) & (seed_delta != 0)
        changed = torch.where(eligible, donor, leader)
    elif family == 'opposed-rollback':
        eligible = (parent - old) * seed_delta < 0
        changed = torch.where(eligible, ancestor, leader)
    elif family == 'scaled-seed':
        eligible = seed_delta != 0
        changed = (parent + spec['factor'] * seed_delta).to(leader.dtype)
    else:
        raise ValueError(f'Unsupported seed edit: {family}')
    return changed, {'eligible': int(eligible.sum())}


def final_channel_tensor(name, leader, spec, sources):
    ancestor, donor = sources
    if name == 'model.language_model.norm.weight':
        assert torch.equal(leader, ancestor.get_tensor(name))
        return donor.get_tensor(name), {'eligible': leader.numel()}
    prefix = f"model.language_model.layers.{spec['layer']}.mlp."
    if not name.startswith(prefix):
        return None
    module = name.removeprefix(prefix)
    if module not in ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight'):
        return None
    channels = spec['channel_indices']
    index = (slice(None), channels) if module == 'down_proj.weight' else (channels, slice(None))
    source = donor.get_tensor(name)
    original = ancestor.get_tensor(name)
    assert int((source != original).sum()) == int((source[index] != original[index]).sum())
    changed = leader.clone()
    changed[index] = source[index]
    return changed, {'eligible': changed[index].numel()}
