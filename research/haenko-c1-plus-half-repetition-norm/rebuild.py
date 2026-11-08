"""Interpolate a hash-verified published edit on a compatible parent.

Adapted from SpacexCeo’s rebuild.py in Delta League PR #24.
Usage: python rebuild.py model/model.safetensors upstream edited.safetensors 0.5 REFERENCE_NORM_MODEL DONOR_NORM_MODEL
Run without -O so the tensor hash assertions remain enabled.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, safe_open, save_file

parent, upstream, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
w = load_file(parent)
manifest = json.loads((upstream / 'manifest.json').read_text())
with np.load(upstream / 'edit.npz', allow_pickle=False) as edits:
    for i, entry in enumerate(manifest['tensors']):
        t = w[entry['name']]
        assert t.dtype == torch.bfloat16 and list(t.shape) == entry['shape']
        bits = t.view(torch.uint16).numpy().copy()
        assert hashlib.sha256(bits.tobytes()).hexdigest() == entry['base_sha256'], 'parent tensor differs from upstream base'
        assert entry['form'] == 'sparse'
        bits.reshape(-1)[edits[f't{i}_idx']] = edits[f't{i}_val']
        assert hashlib.sha256(bits.tobytes()).hexdigest() == entry['edited_sha256']
        edited = torch.from_numpy(bits).view(torch.bfloat16)
        w[entry['name']] = (t.float() + alpha * (edited.float() - t.float())).to(torch.bfloat16)
        print(entry['name'], int((w[entry['name']] != t).sum()), f'entries changed at alpha={alpha}; upstream hashes verified')
norm = 'model.language_model.norm.weight'
with safe_open(sys.argv[5], framework='pt') as reference, safe_open(sys.argv[6], framework='pt') as donor:
    reference_norm = reference.get_tensor(norm)
    donor_norm = donor.get_tensor(norm)
    assert torch.equal(w[norm].view(torch.uint16), reference_norm.view(torch.uint16))
    assert torch.equal((1.05 * (1 + w[norm].float()) - 1).to(w[norm].dtype), donor_norm)
    w[norm] = donor_norm
save_file(w, out, metadata={'format': 'pt'})
print('sha256', hashlib.sha256(open(out, 'rb').read()).hexdigest())
