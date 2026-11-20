import sys, torch
from safetensors import safe_open
from transformers import AutoTokenizer
base, new = sys.argv[1], sys.argv[2]
a = safe_open(base + '/model.safetensors', 'pt'); b = safe_open(new + '/model.safetensors', 'pt')
t = AutoTokenizer.from_pretrained(base)
for k in sorted(a.keys()):
    x = a.get_tensor(k); y = b.get_tensor(k)
    if torch.equal(x, y): continue
    d = y.float() - x.float(); print(k, tuple(x.shape), 'L2', round(d.norm().item(), 4))
    if k.endswith('embed_tokens.weight'):
        X = x.float(); Y = y.float(); mu = X.mean(0)
        for i in (d.abs().sum(1) > 0).nonzero().flatten().tolist():
            A = torch.stack([X[i], mu], 1); sol = torch.linalg.lstsq(A, Y[i][:, None]).solution.flatten()
            res = ((A @ sol).flatten() - Y[i]).norm() / Y[i].norm()
            print(f'  {i:7d} {t.convert_ids_to_tokens(i)!r:22} y = {sol[0]:.4f}*e + {sol[1]:.4f}*mu  rel_res={res:.4f}  norm_ratio={Y[i].norm()/X[i].norm():.4f}')
    elif x.dim() == 1:
        r = (1 + y.float()) / (1 + x.float()); print('  (1+w) ratio mean', r.mean().item(), 'std', r.std().item())
    else:
        print('  rows changed', int((d.abs().sum(1) > 0).sum()))
