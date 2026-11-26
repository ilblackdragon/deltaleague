"""Transfer whole MLP neurons selected by differences between two public models."""
import torch


def graft_channels(name, leader, spec, sources):
    layers = 'model.language_model.layers.'
    if not name.startswith(layers):
        return None
    layer, module = name.removeprefix(layers).split('.', 1)
    if not spec['first_layer'] <= int(layer) <= spec['last_layer']:
        return None
    vectors = {'mlp.gate_proj.weight': 1, 'mlp.up_proj.weight': 1, 'mlp.down_proj.weight': 0}
    if module not in vectors:
        return None
    parent, donor = sources
    counts = []
    for matrix, axis in vectors.items():
        key = f'{layers}{layer}.{matrix}'
        counts.append((parent.get_tensor(key) != donor.get_tensor(key)).sum(dim=axis))
    totals = torch.stack(counts).sum(dim=0)
    indices = torch.argsort(totals, descending=True, stable=True)[:len(totals) // spec['divisor']]
    index = (indices, slice(None)) if vectors[module] == 1 else (slice(None), indices)
    changed = leader.clone()
    changed[index] = donor.get_tensor(name)[index]
    return changed, {'selected_channels': len(indices), 'eligible': changed[index].numel()}
