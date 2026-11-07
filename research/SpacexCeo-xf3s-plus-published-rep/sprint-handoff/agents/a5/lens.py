# logit lens of directions through final norm weight and tied unembedding (new leader weights)
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
D = torch.load('results/dirs.pt')
f = safe_open('nl_model/model.safetensors', 'pt'); tok = AutoTokenizer.from_pretrained('nl_model')
E = f.get_tensor('model.language_model.embed_tokens.weight').float(); nw = 1 + f.get_tensor('model.language_model.norm.weight').float()
def orth(a, b): a = a - (a @ b) * b; return a / a.norm()
for k in ['rp_L23', 'rp_L18', 'cp_L23', 'stuck_p_L23', 'up15', 'up19']:
    r = D[k].float(); r = r / r.norm()
    lg = E @ (r * nw)
    print(k, '+', [tok.decode([t]) for t in torch.topk(lg, 20).indices])
    print(k, '-', [tok.decode([t]) for t in torch.topk(-lg, 20).indices])
r = orth(D['rp_L12'].float() / D['rp_L12'].norm(), D['cp_L12'].float() / D['cp_L12'].norm())
print('cos(rp12perp, rp_L) per layer', [round((r @ D[f'rp_L{L}']).item(), 2) for L in range(24)])
print('cos(rp12perp, cp_L) per layer', [round((r @ D[f'cp_L{L}']).item(), 2) for L in range(24)])
print('cos(up15, up19)', (D['up15'] @ D['up19']).item())
for L in [15, 19, 23]: print('cos up15 vs rp/cp/stuck/ro L', L, [round((D['up15'] @ D[f'{k}_L{L}']).item(), 3) for k in ['rp', 'cp', 'stuck_p', 'ro', 'co']])
