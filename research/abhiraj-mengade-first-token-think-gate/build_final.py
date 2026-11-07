"""Write the candidate checkpoint: gain 3 on the final norm, </think> row + delta, filler rows x0.5.

The checkpoint stores w and the model applies (1 + w), so gain g is new_w = (1 + w) * g - 1.
Only the two tensors change, and everything else is copied byte for byte.
"""
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST, DELTA = sys.argv[1], sys.argv[2], sys.argv[3]
GAIN = 3.0
CLOSE = 248069
FILLER = [3655, 11158, 13428, 13784, 77264, 84485, 85152]
EMB = 'model.language_model.embed_tokens.weight'
NORM = 'model.language_model.norm.weight'

tensors = {}
with safe_open(SRC, framework='pt') as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

E = tensors[EMB]
print('embedding', E.dtype, E.shape, 'norm', tensors[NORM].dtype, tensors[NORM].shape)
d = torch.tensor(np.load(DELTA)).float()
W = E[[CLOSE] + FILLER].float()
W[0] += d
W[1:] *= 0.5
E = E.clone()
E[[CLOSE] + FILLER] = W.to(E.dtype)
tensors[EMB] = E
w = tensors[NORM]
tensors[NORM] = ((1 + w.float()) * GAIN - 1).to(w.dtype)
save_file(tensors, DST, metadata=meta)

with safe_open(SRC, framework='pt') as a, safe_open(DST, framework='pt') as b:
    assert set(a.keys()) == set(b.keys())
    changed = [k for k in a.keys() if not torch.equal(a.get_tensor(k), b.get_tensor(k))]
    print('changed tensors', changed)
    x, y = a.get_tensor(EMB), b.get_tensor(EMB)
    print('changed rows', sorted((x != y).any(1).nonzero().flatten().tolist()))
    got = (y[CLOSE].float() - x[CLOSE].float())
    print('row delta error after rounding: %.5f of ||d|| %.4f' % ((got - d).norm(), d.norm()))
    n0, n1 = a.get_tensor(NORM).float(), b.get_tensor(NORM).float()
    print('applied gain mean %.4f min %.4f max %.4f' % (((1 + n1) / (1 + n0)).mean(), ((1 + n1) / (1 + n0)).min(), ((1 + n1) / (1 + n0)).max()))
