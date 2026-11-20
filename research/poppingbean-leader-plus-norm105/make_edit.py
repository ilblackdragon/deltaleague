"""Stack rank-3's final-norm edit onto the rank-1 leader.

Leader 344c8376 = b8af1dbc + half-strength L15/L19 down_proj edit (PR #24).
27e49fd4        = b8af1dbc + final-norm gain x1.05 (only model.language_model.norm.weight).
The two edits touch disjoint tensors, so the stacked model is the leader with its
final-norm tensor replaced bit-for-bit by 27e49fd4's. No new numbers, no data, no seed.

Usage: python make_edit.py LEADER.safetensors PARENT_b8af1dbc.safetensors X105_27e49fd4.safetensors OUT.safetensors
"""
import hashlib, json, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

NAME = 'model.language_model.norm.weight'
leader_p, parent_p, x105_p, out_p = sys.argv[1:5]
sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
P, X = safe_open(parent_p, 'pt'), safe_open(x105_p, 'pt')
# Precondition: 27e49fd4 differs from b8af1dbc only in the final norm.
changed = [k for k in P.keys() if not torch.equal(P.get_tensor(k), X.get_tensor(k))]
assert changed == [NAME], changed
weights = load_file(leader_p)
# Precondition: the leader's final norm is still the parent's (the leader did not touch it).
assert torch.equal(weights[NAME], P.get_tensor(NAME))
new = X.get_tensor(NAME)
before = sha(weights[NAME]); weights[NAME] = new
save_file(weights, out_p, metadata={'format': 'pt'})
ratio = (1 + new.float()) / (1 + P.get_tensor(NAME).float())
print(json.dumps({'tensor': NAME, 'leader_sha256': before, 'stacked_sha256': sha(new),
                  'gain_vs_b8af1dbc': {'mean': ratio.mean().item(), 'min': ratio.min().item(), 'max': ratio.max().item()}}, indent=1))
