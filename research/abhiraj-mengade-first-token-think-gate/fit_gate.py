"""Fit the minimum-norm change d to the </think> row, as a max-margin separator on the final hidden state.

    minimise  ||d||^2
    s.t.      h.(E_c + d) >= max_other(h) + k_pos / g     at think-open positions
              h.(E_c + d) <= max_other(h) - k_neg / g     at answer positions

Margins are in nats after gain g, since gain multiplies every logit. Solved as squared hinge plus ridge with L-BFGS.
"""
import sys

import numpy as np
import torch

g, k_pos, k_neg = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
out = sys.argv[4]
D = np.load(sys.argv[5] if len(sys.argv) > 5 else 'analysis/gate_data.npz')
P, Pc, Pt = map(lambda k: torch.tensor(D[k]).double(), ('P', 'Pc', 'Pt'))
N, Nc, Nt = map(lambda k: torch.tensor(D[k]).double(), ('N', 'Nc', 'Nt'))
n = len(P)
idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
tr, te = idx[: int(.8 * n)], idx[int(.8 * n):]
nidx = torch.randperm(len(N), generator=torch.Generator().manual_seed(0))
ntr, nte = nidx[: int(.8 * len(N))], nidx[int(.8 * len(N)):]
d = torch.zeros(P.shape[1]).double().requires_grad_()
opt = torch.optim.LBFGS([d], max_iter=800, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
lam = 1e-2


def closure():
    opt.zero_grad()
    a = torch.relu((Pt[tr] - Pc[tr]) + k_pos / g - P[tr] @ d)
    b = torch.relu((Nc[ntr] - Nt[ntr]) + k_neg / g + N[ntr] @ d)
    loss = (a ** 2).mean() + (b ** 2).mean() + lam * (d ** 2).sum()
    loss.backward()
    return loss


opt.step(closure)
d = d.detach()
for name, i, j in (('train', tr, ntr), ('heldout', te, nte)):
    mp = g * (Pc[i] + P[i] @ d - Pt[i])
    mn = g * (Nc[j] + N[j] @ d - Nt[j])
    print(f'{name}: open margin nats min {mp.min():.2f} p5 {mp.quantile(.05):.2f} p50 {mp.median():.2f} frac>3 {(mp > 3).double().mean():.4f}'
          f' | answer margin max {mn.max():.2f} p99.9 {mn.quantile(.999):.2f} frac>-3 {(mn > -3).double().mean():.5f}')
Ec = torch.tensor(D['Ec']).double()
print('||d|| %.4f ||Ec|| %.4f ratio %.3f cos(d,Ec) %.3f' % (d.norm(), Ec.norm(), d.norm() / Ec.norm(), torch.cosine_similarity(d, Ec, dim=0)))
mu = P.mean(0)
print('cos(d, mean open h) %.3f  mean h.d at open %.2f, at answers %.2f' % (torch.cosine_similarity(d, mu, dim=0), (P @ d).mean(), (N @ d).mean()))
for k in sorted(set(D['kinds'].tolist())):
    m = torch.tensor(D['kinds'] == k)
    mp = g * (Pc[m] + P[m] @ d - Pt[m])
    print(f'  {k}: n={int(m.sum())} min margin {mp.min():.2f} p5 {mp.quantile(.05):.2f}')
np.save(out, d.float().numpy())
