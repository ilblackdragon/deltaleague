"""a5 builder: directional (rank-1) projection edit on top of the new leader a9340e14.
For every residual-stream writer W (mlp.down_proj, self_attn.o_proj / linear_attn.out_proj) in the chosen layers:
    W <- (I - alpha r r^T) W        (r unit vector in the 1024-d residual stream; no hooks, no new tensors)
which is exactly 'project alpha of the r component out of that module's output'.
usage: build.py DIRS.pt DIRKEY ALPHA LAYERS(all|comma list) MODS(down,attn) OUT_DIR [PARENT]
Keys, shapes and dtypes are asserted identical to the parent; prints the per-tensor change and sha256."""
import sys, os, shutil, hashlib, json, torch
from safetensors.torch import load_file, save_file
dirsf, key, alpha, layers, mods, out = sys.argv[1:7]
parent = sys.argv[7] if len(sys.argv) > 7 else os.path.expanduser('~/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model')
alpha = float(alpha); mods = mods.split(',')
DD = torch.load(dirsf)
def getdir(L=None):
    # 'rp_L12-cp_L12' = rp_L12 orthogonalised against cp_L12; '{L}' = use the direction of the edited layer itself
    ks = [k.replace('{L}', str(L)) for k in key.split('-')]
    r = DD[ks[0]].float(); r = r / r.norm()
    for o in ks[1:]:
        q = DD[o].float(); q = q / q.norm(); r = r - (r @ q) * q; r = r / r.norm()
    return r
r = getdir(0 if '{L}' in key else None)
W = load_file(os.path.join(parent, 'model.safetensors')); ref = {k: (v.shape, v.dtype) for k, v in W.items()}
NL = 24; lys = range(NL) if layers == 'all' else [int(x) for x in layers.split(',')]
P = 'model.language_model.layers.'
names = []
for L in lys:
    if 'down' in mods: names.append(f'{P}{L}.mlp.down_proj.weight')
    if 'attn' in mods:
        n = f'{P}{L}.self_attn.o_proj.weight'
        names.append(n if n in W else f'{P}{L}.linear_attn.out_proj.weight')
log = {}
if mods == ['shift']:  # constant steering vector alpha*r added to every embedding row (tied LM head: softmax-invariant)
    names = []; E = 'model.language_model.embed_tokens.weight'
    w = W[E].float(); W[E] = (w + alpha * r[None, :]).to(torch.bfloat16)
    log[E] = {'rel_change': ((W[E].float() - w).norm() / w.norm()).item()}
for n in names:
    if '{L}' in key: r = getdir(int(n.split('.')[3]))
    w = W[n].float(); w2 = w - alpha * torch.outer(r, r @ w)
    W[n] = w2.to(torch.bfloat16)
    log[n] = {'rel_change': ((W[n].float() - w).norm() / w.norm()).item()}
assert {k: (v.shape, v.dtype) for k, v in W.items()} == ref
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
h = hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest()
json.dump({'dir': key, 'alpha': alpha, 'layers': layers, 'mods': mods, 'parent': parent, 'sha256': h, 'tensors': log},
          open(out.rstrip('/') + '.build.json', 'w'), indent=1)
print(len(log), 'tensors edited; max rel change', max(v['rel_change'] for v in log.values()), 'sha256', h)
