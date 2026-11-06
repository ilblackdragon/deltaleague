"""Scale Qwen3.5-0.8B output logits by ALPHA via the final RMSNorm gain only.

Qwen3_5RMSNorm computes x_hat * (1 + w). Embeddings are tied, so the logits are
embed @ norm(h); multiplying (1 + w) by ALPHA multiplies every logit by ALPHA.
New gain: w' = bf16(ALPHA * (1 + w) - 1). No data, no fitting, deterministic.

Usage: python make_edit.py PATH/TO/model/model.safetensors PATH/TO/edited.safetensors
"""
import hashlib, json, sys
import torch
from safetensors.torch import load_file, save_file

ALPHA = 1.25
NAME = 'model.language_model.norm.weight'
src, dst = sys.argv[1], sys.argv[2]
weights = load_file(src)
w = weights[NAME]
assert w.dtype == torch.bfloat16 and tuple(w.shape) == (1024,)
sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
new = (ALPHA * (1.0 + w.float()) - 1.0).to(torch.bfloat16)
ratio = (1.0 + new.float()) / (1.0 + w.float())
weights[NAME] = new
save_file(weights, dst, metadata={'format': 'pt'})
print(json.dumps({'tensor': NAME, 'alpha': ALPHA, 'changed_entries': int((new != w).sum()),
                  'original_sha256': sha(w), 'edited_sha256': sha(new),
                  'realized_scale': {'mean': ratio.mean().item(), 'min': ratio.min().item(),
                                     'max': ratio.max().item(), 'std': ratio.std().item()}}, indent=1))
