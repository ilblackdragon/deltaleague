"""Build candidate checkpoints from the untouched parent. Deterministic; no data fitting.
A: dampen filler-token embedding rows (tied input embedding + LM head) by 0.4.
B: A + final RMSNorm gain x1.25 (Qwen3.5 norm is x*(1+w), so w' = 1.25*(1+w) - 1).
"""
import hashlib, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
ws = Path(sys.argv[1] if len(sys.argv) > 1 else 'qwen35')
meta = json.loads((ws/'model/delta-model.json').read_text())
parent = ws/'model/model.safetensors'
h = hashlib.sha256(parent.read_bytes()).hexdigest()
assert h == next(f['sha256'] for f in meta['files'] if f['name'] == 'model.safetensors'), 'parent changed'
FILLER = {13784: 'Wait', 13428: ' Wait', 77264: 'Hmm', 85152: ' Hmm', 11158: 'wait', 3655: ' wait', 84485: ' hmm'}
SCALE, GAIN = 0.4, 1.25
EMB, NORM = 'model.language_model.embed_tokens.weight', 'model.language_model.norm.weight'

def build(out, gain=None):
    w = load_file(str(parent))
    e = w[EMB]; ids = torch.tensor(sorted(FILLER))
    e[ids] = (e[ids].float() * SCALE).to(e.dtype)
    if gain:
        n = w[NORM]; w[NORM] = ((1 + n.float()) * gain - 1).to(n.dtype)
    save_file(w, str(ws/out), metadata={'format': 'pt'})
    # report what changed
    p = load_file(str(parent)); changed = {k: int((p[k] != w[k]).sum()) for k in p if not torch.equal(p[k], w[k])}
    print(out, changed)

build('edited.safetensors')                 # candidate A (primary)
build('edited-B-norm125.safetensors', GAIN)  # candidate B (combo)
