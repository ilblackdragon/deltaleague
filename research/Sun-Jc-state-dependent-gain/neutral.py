"""How much of the final residual stream is softmax-neutral, and how constant is that part?

The final logits are U @ (h / rms(h)) with U = E * (1 + w) (output embedding times final-norm
gain). A direction n is softmax-neutral when U @ n is the same for every token, i.e. when it
has no variance across the rows of U. Removing residual norm that lies in such directions
leaves the logit pattern alone but, through the RMS normalisation, scales it up: a gain.

Reports the spectrum of the token-covariance of U and, for stored final-residual states
(site resid:23), the share of |h|^2 inside the k most neutral directions, split into the part
carried by the mean state (removable by a constant write) and the rest.

usage: python neutral.py FP32_DIR ACTS_DIR [ACTS_DIR ...] [--save OUT.pt]
"""
import glob
import os
import sys

import torch
from safetensors import safe_open

args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
if save:
    args.remove(save)
model_dir = os.path.expanduser(args[0])
with safe_open(glob.glob(os.path.join(model_dir, "*.safetensors"))[0], framework="pt") as f:
    names = list(f.keys())
    E = f.get_tensor(next(n for n in names if n.endswith("embed_tokens.weight"))).double()
    w = f.get_tensor(next(n for n in names if n.endswith("language_model.norm.weight"))).double()
U = E * (1 + w)
mean_row = U.mean(0)
C = (U.T @ U) / U.shape[0] - torch.outer(mean_row, mean_row)  # covariance of the rows of U
lam, Q = torch.linalg.eigh(C)  # ascending: column 0 is the most neutral direction
print(f"U: {tuple(U.shape)}; |mean row| = {mean_row.norm():.3f}; mean |row| = {U.norm(dim=1).mean():.3f}; final-norm gain (1+w): mean {float((1 + w).mean()):.3f}, min {float((1 + w).min()):.3f}, max {float((1 + w).max()):.3f}")
print("sqrt(eigenvalues) of the row covariance (std across tokens of the logit shift per unit-norm write, before the 1/rms factor):")
for k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1023):
    print(f"   #{k:4d}: {lam[k].clamp_min(0).sqrt():.5f}")
u_dir = mean_row / mean_row.norm()
print(f"mean-row direction: std of logit shift per unit write {float((u_dir @ C @ u_dir).sqrt()):.4f} (vs uniform shift {mean_row.norm():.3f})")

X, phase = [], []
for d in args[1:]:
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(d), "acts-*.pt"))):
        blob = torch.load(path)
        if "resid:23" not in blob["X"]:
            continue
        X.append(blob["X"]["resid:23"].double())
        phase.append(blob["meta"][:, 2] >= 0)  # True: the state is after the first </think>
X, phase = torch.cat(X), torch.cat(phase)
print(f"\nfinal-residual states: {int((~phase).sum())} thinking, {int(phase.sum())} answer")
for name, sel in (("thinking", ~phase), ("answer", phase)):
    H = X[sel]
    if not len(H):
        continue
    mu = H.mean(0)
    e_total = float(H.pow(2).sum(1).mean())
    norms = H.norm(dim=1)
    print(f"\n{name}: |h| mean {norms.mean():.2f} (cv {norms.std() / norms.mean():.3f}), rms per dim {float((e_total / 1024) ** 0.5):.3f}; "
          f"|mean state|^2 / E|h|^2 = {float(mu.pow(2).sum()) / e_total:.3f}")
    Hq, muq = H @ Q, mu @ Q
    print(f"   {'k':>5s} {'share of |h|^2 in the k most neutral dirs':>42s} {'of which constant (mean state)':>32s} {'gain if constant part removed':>30s} {'logit std of that removal':>26s}")
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        share = float(Hq[:, :k].pow(2).sum(1).mean()) / e_total
        const = float(muq[:k].pow(2).sum()) / e_total
        # removing c = P_k mu from every state: per-state gain |h| / |h - c|, and the logit pattern added is -U c / rms(h - c)
        c = Q[:, :k] @ muq[:k]
        new_norm = (H - c).norm(dim=1)
        g = norms / new_norm
        dmg = float((c @ C @ c).sqrt()) / float((new_norm / 32.0).mean())
        print(f"   {k:5d} {share:42.3f} {const:32.3f} {float(g.mean()):22.3f} +-{float(g.std()):.3f} {dmg:26.3f}")
if save:
    torch.save({"lam": lam, "Q": Q, "mean_row": mean_row}, save)
