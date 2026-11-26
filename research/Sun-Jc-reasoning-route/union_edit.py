"""Copy the union of registered tensor masks from one verified donor."""
def matches(layer, module, spec):
    return spec['first_layer'] <= layer <= spec['last_layer'] and module in spec['modules']


def graft_union(name, leader, spec, sources):
    prefix = 'model.language_model.layers.'
    if not name.startswith(prefix):
        return None
    number, module = name.removeprefix(prefix).split('.', 1)
    if not any(matches(int(number), module, part) for part in spec['pieces']):
        return None
    donor = sources[1].get_tensor(name)
    assert donor.shape == leader.shape and donor.dtype == leader.dtype
    return donor, {'source_differences': int((donor != leader).sum())}
