"""Design and offline check of a state-dependent logit gain built from the neutral coordinate.

The edit (all in the last layer, so it changes output distributions but never the model's
internal state):

  final norm   every coordinate's gain (1 + w_j) is multiplied by G, except coordinate 0.
  coordinate 0 is the model's softmax-neutral "norm reservoir": the output embedding barely
               varies along it, so writing into it changes no logit differences, only the RMS
               that the final norm divides by. Its own norm gain is lowered so that its small
               logit contribution stays what it was on average.
  neuron A     always on. Writes a(x) into coordinate 0, where a(x) is a linear read of the MLP
               input fitted so that |h + a e0| = (G / G0) |h|: the RMS normalisation then turns
               the gain G into G0 (default 1: no change).
  neurons B, C a soft step on a linear feature that separates the "gain class" from the "keep
               class". In gain-class states they remove what neuron A wrote, so the gain G
               acts. In keep-class states both are exponentially close to zero.

Classes are lists of family:phase with phase in {thinking, answer, all}; "*" stands for every
family that is not measure-only. Two useful settings:
  answer writing   --gain-class prose:answer,short:answer,struct:answer,code:answer --keep-class "*:thinking"
  prose-type task  --gain-class prose:all --keep-class math:all,gsm8k:all,arc:all,csqa:all,mmlu:all

The script fits the three reads on stored states (sites mlp_in:23 and resid:23), picks
expendable neurons, simulates the edited last layer on held-out completions and prints the
effective gain per population. Families named in --measure-only are never used for fitting.
With --save it writes the edit records for edits.py.

usage: python phase_gain.py --model FP32_DIR --acts ACTS_DIR [ACTS_DIR ...] --gain-class SPEC --keep-class SPEC
                            [--gain 1.25] [--keep-gain 1.0] [--beta 12] [--measure-only ifeval] [--save EDITS.pt]
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

import edits as edit_lib

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--acts", nargs="+", required=True)
ap.add_argument("--gain", type=float, default=1.25, help="logit gain in gain-class states")
ap.add_argument("--keep-gain", type=float, default=1.0, help="net logit gain in keep-class states")
ap.add_argument("--gain-class", default="prose:answer,short:answer,struct:answer,code:answer")
ap.add_argument("--keep-class", default="*:thinking")
ap.add_argument("--beta", type=float, default=12.0, help="steepness of the step, in units of half the gap between the class means")
ap.add_argument("--shift", type=float, default=0.0, help="shift of the step threshold in the same units: positive gates more states, negative fewer")
ap.add_argument("--const", type=float, default=8.0, help="gate value of the always-on neuron (linear regime of SiLU)")
ap.add_argument("--measure-only", default="ifeval", help="families excluded from every fit")
ap.add_argument("--keep", default="1998,3529", help="neurons that must not be overwritten (inherited grafts)")
ap.add_argument("--neurons", default="", help="three neuron indices to overwrite (default: the least active ones)")
ap.add_argument("--save", default="")
args = ap.parse_args()
G_TARGET, G_KEEP = args.gain, args.keep_gain
RATIO = G_TARGET / G_KEEP
PREFIX = "model.language_model."

with safe_open(os.path.join(os.path.expanduser(args.model), "model.safetensors"), framework="pt") as f:
    Wg, Wu, Wd = (f.get_tensor(PREFIX + f"layers.23.mlp.{n}.weight").float() for n in ("gate_proj", "up_proj", "down_proj"))
    norm_w = f.get_tensor(PREFIX + "norm.weight").float()
    E = f.get_tensor(PREFIX + "embed_tokens.weight")

xs, hs, metas, rows = [], [], [], []
for d in args.acts:
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(d), "acts-*.pt"))):
        part = torch.load(path)
        m = part["meta"].clone()
        m[:, 0] += len(rows)
        metas.append(m)
        rows.extend(part["rows"])
        xs.append(part["X"]["mlp_in:23"].float())
        hs.append(part["X"]["resid:23"].float())
X, H, meta = torch.cat(xs), torch.cat(hs), torch.cat(metas)
del xs, hs
row, step, since, total = meta[:, 0].long(), meta[:, 1], meta[:, 2], meta[:, 4]
fam = [rows[int(i)]["family"] for i in row]
is_fam = lambda *names: torch.tensor([f in names for f in fam])
ok = torch.tensor([rows[int(i)]["outcome"] == "ok" for i in row])
think, tail = since < 0, total - step
measure_only = is_fam(*args.measure_only.split(","))
boxed = is_fam("math", "gsm8k", "arc", "csqa", "mmlu")
families = sorted(set(fam))


def class_mask(spec):
    """States selected by a family:phase list ("*" = every family that is not measure-only). Answers: finished completions only."""
    mask = torch.zeros(len(X), dtype=torch.bool)
    for item in spec.split(","):
        name, _, phase = item.partition(":")
        f = ~measure_only if name == "*" else is_fam(name)
        mask |= f & {"thinking": think, "answer": ~think & ok, "all": think | ok}[phase or "all"]
    return mask


gain_class, keep_class = class_mask(args.gain_class), class_mask(args.keep_class)
keep_class &= ~gain_class
pops = {}
for name in families:
    pops[f"{name}, thinking"] = think & is_fam(name)
    pops[f"{name}, answer"] = ~think & is_fam(name) & ok
pops["boxed families, last 16 answer tokens"] = ~think & boxed & ok & (tail <= 16)
pops["any family, capped completion, after </think>"] = ~think & ~ok
n_rows = len(rows)
perm = torch.randperm(n_rows, generator=torch.Generator().manual_seed(0))
held_rows = torch.zeros(n_rows, dtype=torch.bool)
held_rows[perm[: n_rows // 4]] = True
held = held_rows[row] | measure_only  # never fitted on
train = ~held
import collections as _c
print("completions by family:", dict(sorted(_c.Counter(r["family"] for r in rows).items())))
print(f"{len(X)} states from {n_rows} completions; fitting on {int(train.sum())} states, {int(held.sum())} held out "
      f"(a quarter of the completions, and every state of: {args.measure_only})")


def moments(mask):
    """Mean and covariance of the MLP input over the selected states, in float64."""
    x = X[mask].double()
    return x.mean(0), torch.cov(x.T)


# --- expendable neurons: smallest largest-ever write into the residual stream --------------
keep = {int(i) for i in args.keep.split(",") if i}
col_norm = Wd.norm(dim=0)
max_write, sq_write = torch.zeros(Wg.shape[0]), torch.zeros(Wg.shape[0])
for a in range(0, len(X), 4096):
    x = X[a:a + 4096]
    act = (F.silu(x @ Wg.T) * (x @ Wu.T)).abs() * col_norm
    max_write = torch.maximum(max_write, act.max(0).values)
    sq_write += act.pow(2).sum(0)
rms_write = (sq_write / len(X)).sqrt()
order = [int(i) for i in torch.argsort(max_write) if int(i) not in keep]
print(f"layer-23 MLP: {Wg.shape[0]} neurons; |h| mean {float(H.norm(dim=1).mean()):.2f}. Least active (largest write norm over all states / rms):")
for i in order[:6]:
    print(f"   neuron {i:4d}: max {float(max_write[i]):.4f}  rms {float(rms_write[i]):.5f}")
print(f"   median neuron: max {float(max_write.median()):.3f}  rms {float(rms_write.median()):.4f}")
neurons = [int(i) for i in args.neurons.split(",")] if args.neurons else order[:3]

# --- the three linear reads ---------------------------------------------------------------
eye = torch.eye(1024, dtype=torch.float64)
mu, cov = moments(train)
kappa = torch.linalg.solve(cov + 1e-2 * eye, mu)
kappa = kappa / (kappa @ mu)  # a linear functional that is 1 on average, with small variance
k_all = X @ kappa.float()
print(f"\nconstant read: sd {float(k_all[held].std()):.4f} on held-out states (min {float(k_all[held].min()):.3f}, max {float(k_all[held].max()):.3f})")

pos, neg = keep_class & train, gain_class & train
(mu_p, cov_p), (mu_n, cov_n) = moments(pos), moments(neg)
d = torch.linalg.solve(cov_p + cov_n + 1e-3 * eye, mu_p - mu_n)
mid, half = (mu_p @ d + mu_n @ d) / 2, (mu_p @ d - mu_n @ d) / 2
phase = (d - (mid + args.shift * half) * kappa) / half  # +1 keep-class mean, -1 gain-class mean (before the shift)
print(f"class feature fitted on {int(pos.sum())} keep-class states ({args.keep_class}) vs {int(neg.sum())} gain-class states ({args.gain_class})")

sign = 1.0 if float(H[think, 0].mean()) > 0 else -1.0
h0, rho = sign * H[:, 0], H.norm(dim=1)
a_star = -h0 + torch.sqrt(h0 ** 2 + (RATIO ** 2 - 1) * rho ** 2)
fit = keep_class & train
feats = (X[fit] * k_all[fit, None]).double()  # neuron A computes silu(C * kappa.x) * (up.x) ~ C * (kappa.x) * (up.x)
read = torch.linalg.solve(feats.T @ feats + 10.0 * eye, feats.T @ a_star[fit].double())
del feats
check = keep_class & held
pred = (X[check] * k_all[check, None]) @ read.float()
print(f"write needed to cancel the gain: mean {float(a_star[keep_class].mean()):.2f}, sd {float(a_star[keep_class].std()):.2f}; "
      f"held-out R2 of the linear read {1 - float(((pred - a_star[check]) ** 2).mean() / a_star[check].var()):.3f} ({int(check.sum())} keep-class states)")

C = args.const
gate_rows = torch.stack([C * kappa, -args.beta * phase + 0.5 * kappa, -args.beta * phase - 0.5 * kappa]).float()
up_rows = torch.stack([read / C, read, read]).float()
down_cols = torch.zeros(1024, 3)
down_cols[0, 0], down_cols[0, 1], down_cols[0, 2] = sign, -sign, sign
gate_rows, up_rows, down_cols = edit_lib.bf16(gate_rows), edit_lib.bf16(up_rows), edit_lib.bf16(down_cols)
print(f"reads: |gate A| {float(gate_rows[0].norm()):.2f}, |gate B| {float(gate_rows[1].norm()):.2f}, |up| {float(up_rows[1].norm()):.2f}; "
      f"existing rows: |gate| median {float(Wg.norm(dim=1).median()):.2f}, |up| median {float(Wu.norm(dim=1).median()):.2f}; overwriting neurons {neurons}")


def simulate(sel, bf16_acts=False):
    """Final residual after the edit for the selected states."""
    x = X[sel]
    r = edit_lib.bf16 if bf16_acts else (lambda t: t)
    new = r(r(F.silu(r(x @ gate_rows.T))) * r(x @ up_rows.T))
    old = F.silu(x @ Wg[neurons].T) * (x @ Wu[neurons].T)
    return H[sel] + new @ down_cols.T - old @ Wd[:, neurons].T


rms = lambda h: h.pow(2).mean(-1).sqrt()
# --- final norm: gain G everywhere except coordinate 0 -----------------------------------
one_plus = 1 + norm_w
sel_t = torch.nonzero(keep_class & train)[:, 0]
h_new_t = simulate(sel_t)
ratio0 = float((H[sel_t, 0] / rms(H[sel_t])).mean() / (h_new_t[:, 0] / rms(h_new_t)).mean())
new_one_plus = one_plus * G_TARGET
new_one_plus[0] = one_plus[0] * ratio0
new_norm_w = edit_lib.bf16(new_one_plus - 1)
print(f"final norm: (1 + w) x {G_TARGET} on 1,023 coordinates; coordinate 0: {float(one_plus[0]):.3f} -> {float(1 + new_norm_w[0]):.4f}")


def logits(h, w):
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)) @ E.float().T


# --- effective gain per population, on states never used for fitting -------------------------
print(f"\neffective logit gain after the edit (target {G_TARGET} in the gain class, {G_KEEP} in the keep class), on held-out states. 'bf16': activations rounded to bfloat16.")
print("pattern: sd over the parent's top-64 tokens of (edited logits - gain x parent logits); tv: total variation between the parent's and the edited next-token distribution")
print(f"   {'population':46s} {'n':>6s} {'mean':>6s} {'sd':>6s} {'q01':>6s} {'q05':>6s} {'q50':>6s} {'q95':>6s} {'q99':>6s} {'class':>6s} {'gated':>6s} {'bf16 mean/sd':>13s}  {'per-completion mean':>20s} {'pattern':>8s} {'tv':>6s}")
for name, mask in pops.items():
    sel = torch.nonzero(mask & held)[:, 0]
    if len(sel) < 20:
        continue
    h_new = simulate(sel)
    g_eff = G_TARGET * rms(H[sel]) / rms(h_new)
    g_bf = G_TARGET * rms(H[sel]) / rms(simulate(sel, bf16_acts=True))
    q = torch.quantile(g_eff, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    cnt = torch.bincount(row[sel], minlength=n_rows).float()
    per = (torch.zeros(n_rows).index_add_(0, row[sel], g_eff) / cnt.clamp_min(1))[cnt >= 20]
    sub = sel[:: max(1, len(sel) // 300)]
    z_old, z_new = logits(H[sub], norm_w), logits(simulate(sub), new_norm_w)
    top = z_old.topk(64, dim=-1).indices
    g_sub = (G_TARGET * rms(H[sub]) / rms(simulate(sub)))[:, None]
    err = z_new.gather(1, top) - g_sub * z_old.gather(1, top)
    err = err - err.mean(1, keepdim=True)
    tv = 0.5 * (torch.softmax(z_new, -1) - torch.softmax(z_old, -1)).abs().sum(-1)
    print(f"   {name:46s} {len(sel):6d} {float(g_eff.mean()):6.3f} {float(g_eff.std()):6.3f} " + " ".join(f"{float(v):6.3f}" for v in q)
          + f" {('gain' if float(gain_class[sel].float().mean()) > 0.5 else 'keep' if float(keep_class[sel].float().mean()) > 0.5 else '-'):>6s} {float((g_eff > (G_TARGET * G_KEEP) ** 0.5).float().mean()):6.3f} {float(g_bf.mean()):6.3f}/{float(g_bf.std()):5.3f}"
          + (f"  {float(per.min()):.3f}..{float(per.max()):.3f} (n={len(per):3d})" if len(per) else f"  {'-':>20s}")
          + f" {float(err.std()):8.4f} {float(tv.mean()):6.3f}")

if args.save:
    records = []
    for j, i in enumerate(neurons):
        records += edit_lib.neuron(23, i, gate_rows[j], up_rows[j], down_cols[:, j])
    records.append(("norm.weight", "all", None, new_norm_w))
    edit_lib.save(records, args.save, note=f"gain {G_TARGET} on [{args.gain_class}], {G_KEEP} on [{args.keep_class}]: neurons {neurons}, beta {args.beta}, shift {args.shift}")
    print(f"\nsaved {len(records)} edit records to {args.save}")
