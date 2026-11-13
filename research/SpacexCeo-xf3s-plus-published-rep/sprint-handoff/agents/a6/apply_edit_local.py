# a6 local copy of ~/dl/apply_edit.py (fetched with gpu.py get dl/apply_edit.py, 2026-11-05 17:50); logic unchanged.
"""Deterministic weight edit for Qwen3.5-0.8B (Delta League). No randomness; no training.
Edits (all relative to the ORIGINAL baseline 0ea62f40 so the leader is one point of the family):
  * embedding rows (tied with lm_head): e_t <- s*e_t + (1-s)*mean(E)   (mode 'mean', what the leader used)
                                     or e_t <- s*e_t                   (mode 'scale')
  * final RMSNorm gain: (1+w) <- g*(1+w)
  * optional additive row shift along mean direction (bias): e_t <- e_t + b*mean(E)/|mean(E)|^2 * c   (logit bias ~ b on avg)
  * optional published repetition-direction edit scaled by alpha (delta from upstream edit.npz)
usage: apply_edit.py SPEC.json   (spec: base, out, tokens{str:s} (mode mean|scale), scale_tokens{str:s} (always pure scale),
       close_gain (scales '</think>' row, or close_tokens list), gain, bias{str:b}, rep_alpha, rep_dir)
Prints sha256 of the output weights and L2 of the change per touched tensor."""
import json, sys, os, hashlib, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
spec = json.load(open(sys.argv[1]))
base = spec['base']; out = spec['out']
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):
    if f != 'model.safetensors' and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), os.path.join(out, f))
W = load_file(os.path.join(base, 'model.safetensors'))
orig = {}
EMB = 'model.language_model.embed_tokens.weight'; NORM = 'model.language_model.norm.weight'
tok = AutoTokenizer.from_pretrained(base)
vocab = tok.get_vocab()
def ids_for(s):
    """exact single-token match of the string (Ġ marks a leading space)"""
    s2 = s.replace(' ', 'Ġ')
    if s2 not in vocab: raise SystemExit(f'not a single token: {s!r}')
    return vocab[s2]
E = W[EMB].float(); orig[EMB] = W[EMB].clone()
mu = E.mean(0)
mode = spec.get('mode', 'mean')
touched = []
for s, sc in spec.get('tokens', {}).items():
    i = ids_for(s); touched.append((s, i, sc))
    E[i] = sc * E[i] + (1 - sc) * mu if mode == 'mean' else sc * E[i]
# pure row scaling regardless of mode (new leader a9340e14 uses e<-0.5e for 7 Wait/Hmm rows and e<-1.3e for </think>)
for s, sc in spec.get('scale_tokens', {}).items():
    i = ids_for(s); touched.append((s, i, f'scale {sc}')); E[i] = sc * E[i]
cg = spec.get('close_gain', 1.0)
if cg != 1.0:
    for s in spec.get('close_tokens', ['</think>']):
        i = ids_for(s); touched.append((s, i, f'close_gain {cg}')); E[i] = cg * E[i]
for s, b in spec.get('bias', {}).items():
    i = ids_for(s); E[i] = E[i] + b * mu / mu.dot(mu)
W[EMB] = E.to(torch.bfloat16)
g = spec.get('gain', 1.0)
if g != 1.0:
    orig[NORM] = W[NORM].clone()
    W[NORM] = (g * (1 + W[NORM].float()) - 1).to(torch.bfloat16)
ra = spec.get('rep_alpha', 0)
if ra:
    import numpy as np
    man = json.load(open(os.path.join(spec['rep_dir'], 'manifest.json')))
    with np.load(os.path.join(spec['rep_dir'], 'edit.npz'), allow_pickle=False) as ed:
        for k, entry in enumerate(man['tensors']):
            name = entry['name']; o = W[name]; orig.setdefault(name, o.clone())
            bits = o.view(torch.uint16).numpy().copy().reshape(-1)
            bits[ed[f't{k}_idx']] = ed[f't{k}_val']
            edited = torch.from_numpy(bits.reshape(o.shape)).view(torch.bfloat16).float()
            W[name] = (o.float() + ra * (edited - o.float())).to(torch.bfloat16)
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
h = hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest()
print('tokens', [(s, i, sc) for s, i, sc in touched])
for k, v in orig.items(): print('L2', k, (W[k].float() - v.float()).norm().item())
print('sha256', h)
