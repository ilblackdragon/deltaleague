"""TIES merge without trimming on a shared base: per element, elect sign = sign(sum of source deltas); merged delta = mean of the
source deltas that have the elected sign (sources that did not change the element do not vote). float64 math, stored in the base dtype.
The final norm is taken from NORM_SRC. usage: ties.py BASE.safetensors NORM_SRC.safetensors OUT.safetensors SRC1 SRC2 [SRC3 ...]
A source may be 'patch:<submission_dir>' (reconstructed against BASE)."""
import sys, json, torch
from safetensors import safe_open
from safetensors.torch import save_file
base, normsrc, out, srcs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
B = safe_open(base, 'pt'); N = safe_open(normsrc, 'pt'); S = []
for s in srcs:
    if s.startswith('patch:'):
        from delta_cli.artifacts import reconstruct
        d = s[6:]; S.append(reconstruct(base, f'{d}/weights.safetensors', json.load(open(f'{d}/manifest.json')), allowed_prefix='model.language_model.'))
    else: S.append(safe_open(s, 'pt'))
get = lambda src, k: src[k] if isinstance(src, dict) else src.get_tensor(k)
res = {}; NK = 'model.language_model.norm.weight'
for k in B.keys():
    b = B.get_tensor(k)
    if k == NK: res[k] = N.get_tensor(k); continue
    xs = [get(s, k) for s in S]
    if all(torch.equal(x, b) for x in xs): res[k] = b; continue
    outt = torch.empty_like(b); n = b.shape[0] if b.dim() > 0 else 1; step = max(1, (1 << 22) // max(1, b[0].numel() if b.dim() > 0 else 1))
    for i in range(0, n, step):
        sl = slice(i, i + step) if b.dim() > 0 else slice(None)
        bb = b[sl].double(); D = torch.stack([x[sl].double() - bb for x in xs]); sgn = torch.sign(D.sum(0))
        keep = (torch.sign(D) == sgn) & (D != 0); cnt = keep.sum(0).clamp(min=1)
        outt[sl] = (bb + (D * keep).sum(0) / cnt).to(b.dtype)
        if b.dim() == 0: break
    res[k] = outt.contiguous()
save_file(res, out, metadata={'format': 'pt'}); print('merged', len(S), 'sources')
