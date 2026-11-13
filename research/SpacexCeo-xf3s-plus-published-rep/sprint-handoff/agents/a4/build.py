"""Conditional-closing builder: e_close <- c * e_close + beta * u   (u: direction in final-norm-output space, .npy)
Because the head is tied and the final norm output hn already includes (1+w)*gain, the </think> logit changes by
  (c-1) * hn.e_close + beta * hn.u
so with u = d/|d|^2 the boost is beta * (hn.d)/|d|^2, active only where hn aligns with d.
usage: build.py PARENT_DIR OUT_DIR U.npy BETA [C=1.0] [OFFSET=0.0]
  OFFSET b additionally adds b*mu/|mu|^2 (the queue's 'bias' convention) to cancel a mean shift.
Only row 248069 of embed_tokens changes; all other tensors are copied bit-exactly."""
import sys, os, shutil, hashlib, numpy as np, torch
from safetensors.torch import load_file, save_file
parent, out, upath, beta = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
c = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
off = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
W = load_file(os.path.join(parent, 'model.safetensors'))
EMB = 'model.language_model.embed_tokens.weight'; CLOSE = 248069
E = W[EMB]; e0 = E[CLOSE].float().clone()
u = torch.tensor(np.load(upath)).float()
mu = E.float().mean(0)
new = c * e0 + beta * u + off * mu / mu.dot(mu)
E[CLOSE] = new.to(torch.bfloat16)
W[EMB] = E
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
realised = E[CLOSE].float() - e0
print('row L2 before', e0.norm().item(), 'delta L2 wanted', (new - e0).norm().item(), 'realised', realised.norm().item(),
      'cos', torch.nn.functional.cosine_similarity(realised, new - e0, 0).item())
print('sha256', hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest())
