"""Interpolate a hash-verified published edit on a compatible parent.

Adapted from SpacexCeo’s rebuild.py in Delta League PR #24.
Usage: python rebuild.py model/model.safetensors upstream edited.safetensors 0.5
Run without -O so the tensor hash assertions remain enabled.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file

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
save_file(w, out, metadata={'format': 'pt'})
print('sha256', hashlib.sha256(open(out, 'rb').read()).hexdigest())
