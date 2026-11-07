"""Stack our x1.05 final-norm step (cd0b95bb, #43) onto the GRPO leader 9492e219.

9492e219 = 344c8376 + GRPO full fine-tune (20 steps); it leaves the final norm unchanged.
cd0b95bb = 344c8376 + final-norm gain x1.05 (only model.language_model.norm.weight).
The stacked model is 9492e219 with its final-norm tensor replaced bit-for-bit by
cd0b95bb's. No new numbers, no data, no seed.

Usage: python make_edit.py GRPO_9492e219.safetensors PARENT_344c8376.safetensors X105_cd0b95bb.safetensors OUT.safetensors
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
                  'gain_vs_344c8376': {'mean': ratio.mean().item(), 'min': ratio.min().item(), 'max': ratio.max().item()}}, indent=1))
