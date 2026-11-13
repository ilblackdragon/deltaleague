"""r2b: group-sharpening of answer-token rows on b8af1dbc (tied embeddings = LM head).
For a token group G with rows e_t: e_t <- m_G + s (e_t - m_G), m_G = mean of the group's rows (fp32, cast back to bf16).
Logit effect: z_t = h.m_G + s (h.e_t - h.m_G): within-group logit differences x s (= temperature 1/s among the group),
group-mean logit unchanged. Everything else bit-identical.
usage: build.py OUT_DIR GROUP:S [GROUP:S ...]   groups: digits ('0'..'9', ids 15-24), letters ('A'..'J', ids 32-41)"""
import sys, os, json, shutil, hashlib, torch
from safetensors.torch import load_file, save_file
GROUPS = {'digits': list(range(15, 25)), 'letters': list(range(32, 42))}
out = sys.argv[1]; specs = [(a.split(':')[0], float(a.split(':')[1])) for a in sys.argv[2:]]
parent = os.path.expanduser('~/dl/versions/b8af1dbc-d52b-4bbc-a302-3505d897303c/model')
W = load_file(os.path.join(parent, 'model.safetensors')); ref = {k: (v.shape, v.dtype) for k, v in W.items()}
K = 'model.language_model.embed_tokens.weight'; E = W[K].clone(); log = {}
for g, s in specs:
    ids = torch.tensor(GROUPS[g]); X = E[ids].float(); m = X.mean(0, keepdim=True)
    Y = (m + s * (X - m)).to(E.dtype); E[ids] = Y
    log[g] = {'s': s, 'ids': GROUPS[g], 'row_norm_before': X.norm(dim=1).tolist(), 'row_norm_after': Y.float().norm(dim=1).tolist()}
W[K] = E.contiguous()
assert {k: (v.shape, v.dtype) for k, v in W.items()} == ref
changed = (W[K] != load_file(os.path.join(parent, 'model.safetensors'))[K]).any(1).nonzero().flatten().tolist()
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
h = hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest()
json.dump({'parent': parent, 'groups': log, 'changed_rows': changed, 'sha256': h}, open(out.rstrip('/') + '.build.json', 'w'), indent=1)
print('changed rows', changed, 'sha256', h)
