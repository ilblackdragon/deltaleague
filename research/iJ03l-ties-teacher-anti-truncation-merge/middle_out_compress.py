import argparse
import gc
import math
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def middle_out_weight(layer_idx: int, center: float = 20.0, sigma: float = 3.5, l_min: int = 10, l_max: int = 22) -> float:
    """Compute Gaussian middle-out compression depth weight for a transformer layer."""
    if layer_idx < l_min or layer_idx > l_max:
        return 0.0
    return math.exp(-((layer_idx - center) ** 2) / (2.0 * (sigma ** 2)))


def wiener_spectral_compress(weight: torch.Tensor, strength: float, power: int = 4, preserve_norm: bool = False) -> tuple[torch.Tensor, float, float]:
    """
    Apply smooth high-order spectral Wiener shrinkage to a 2D weight matrix:
        s_new = s * (s^p) / (s^p + (strength * s_med)^p)
    Returns (compressed_matrix, retained_energy_ratio, effective_rank_reduction_ratio).
    """
    if strength <= 0.0:
        return weight, 1.0, 1.0

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    s_med = S.median()
    cutoff = strength * s_med
    sp = S.pow(power)
    cp = cutoff.pow(power)
    shrink = sp / (sp + cp + 1e-12)
    S_new = S * shrink

    total_e = (S ** 2).sum().item()
    kept_e = (S_new ** 2).sum().item()
    if preserve_norm and kept_e > 0:
        S_new = S_new * math.sqrt(total_e / kept_e)

    sr_old = ((S ** 2).sum() / (S[0] ** 2 + 1e-12)).item()
    sr_new = ((S_new ** 2).sum() / (S_new[0] ** 2 + 1e-12)).item()

    recon = (U * S_new) @ Vh
    del U, S, Vh, sp, cp, shrink, S_new
    return recon, kept_e / (total_e + 1e-12), sr_new / (sr_old + 1e-12)


def compute_answer_coord_boost(emb: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute an output-only per-coordinate gain vector [hidden_size] aligned with
    answer tokens (digits 0-9, choices A-E, True/False) relative to general vocabulary.
    """
    if eps == 0.0:
        return torch.ones(emb.shape[1], dtype=torch.float32)

    ans_ids = list(range(15, 25)) + list(range(32, 37)) + [2434, 3913]
    gen = torch.Generator().manual_seed(42)
    rand_idx = torch.randperm(emb.shape[0], generator=gen)[:8000]

    a_energy = emb[ans_ids].float().pow(2).mean(dim=0)
    r_energy = emb[rand_idx].float().pow(2).mean(dim=0)
    selectivity = torch.log((a_energy + 1e-8) / (r_energy + 1e-8))
    # Zero-mean clamped selectivity so overall RMSNorm temperature stays unchanged
    sel_centered = torch.clamp(selectivity - selectivity.mean(), min=-1.5, max=1.5)
    return 1.0 + eps * sel_centered


def main() -> None:
    parser = argparse.ArgumentParser(description="Middle-Out Spectral Wiener Compression + Decoupled Output Calibration")
    parser.add_argument("--base", default="/home/x-5194/NearRadar/experiment/model/model.safetensors")
    parser.add_argument("--output", default="/home/x-5194/NearRadar/experiment/edited_middle_out.safetensors")
    parser.add_argument("--lambda-wiener", type=float, default=0.85, help="Peak Wiener threshold multiplier at middle-out center")
    parser.add_argument("--wiener-power", type=int, default=4, help="Spectral filter sharpness exponent")
    parser.add_argument("--center", type=float, default=20.0, help="Center layer of Middle-Out compression envelope")
    parser.add_argument("--sigma", type=float, default=3.5, help="Gaussian width sigma of Middle-Out envelope")
    parser.add_argument("--norm-gain", type=float, default=1.010, help="Global final RMSNorm gain")
    parser.add_argument("--ans-eps", type=float, default=0.035, help="Output-only answer coordinate sharpening strength")
    parser.add_argument("--l23-kappa", type=float, default=1.18, help="Symmetric boost on Layer 23 graft neurons 1998 & 3529")
    parser.add_argument("--compress-attn", action="store_true", help="Also compress attention out_proj / o_proj in middle-out window")
    parser.add_argument("--preserve-norm", action="store_true", help="Restore exact Frobenius norm after Wiener shrinkage")
    args = parser.parse_args()

    if not os.path.exists(args.base):
        print(f"Missing base checkpoint: {args.base}", file=sys.stderr)
        sys.exit(1)

    print(f"Middle-Out Compression config: lambda={args.lambda_wiener}, p={args.wiener_power}, center={args.center}, sigma={args.sigma}, "
          f"norm={args.norm_gain}, ans_eps={args.ans_eps}, l23_kappa={args.l23_kappa}, "
          f"attn={args.compress_attn}, preserve_norm={args.preserve_norm}")

    merged = {}
    with safe_open(args.base, framework="pt") as f:
        keys = list(f.keys())
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
        ans_boost = compute_answer_coord_boost(emb, args.ans_eps)
        del emb

        for idx, k in enumerate(keys):
            t = f.get_tensor(k)
            is_mlp_down = ".mlp.down_proj.weight" in k
            is_attn_out = args.compress_attn and (".linear_attn.out_proj.weight" in k or ".self_attn.o_proj.weight" in k)

            if is_mlp_down or is_attn_out:
                layer_idx = int(k.split(".layers.")[1].split(".")[0])
                w = middle_out_weight(layer_idx, center=args.center, sigma=args.sigma)
                if w > 0.0:
                    comp, e_ratio, sr_ratio = wiener_spectral_compress(
                        t.float(), args.lambda_wiener * w, power=args.wiener_power, preserve_norm=args.preserve_norm
                    )
                    merged[k] = comp.to(t.dtype)
                    kind = "down_proj" if is_mlp_down else "attn_out "
                    print(f"  L{layer_idx:02d} {kind} (w={w:.2f}): kept_energy={e_ratio*100:.2f}%, stable_rank_ratio={sr_ratio:.3f}")
                    del comp
                else:
                    merged[k] = t
            else:
                merged[k] = t

            del t
            if (idx + 1) % 80 == 0:
                gc.collect()

    # Apply calibrated final RMSNorm + output-only answer subspace boost (leaving embed_tokens pure)
    norm_key = "model.language_model.norm.weight"
    if norm_key in merged:
        nw = merged[norm_key].float() * args.norm_gain * ans_boost
        merged[norm_key] = nw.to(merged[norm_key].dtype)
        print(f"Applied final RMSNorm gain={args.norm_gain} with output-only answer selectivity (ans_eps={args.ans_eps}, boost range [{ans_boost.min():.3f}, {ans_boost.max():.3f}])")

    # Apply symmetric calibration to true Dawgster Layer 23 graft neurons 1998 & 3529
    down23_key = "model.language_model.layers.23.mlp.down_proj.weight"
    if down23_key in merged and args.l23_kappa != 1.0:
        d23 = merged[down23_key].clone()
        d23[:, 1998] = d23[:, 1998] * args.l23_kappa
        d23[:, 3529] = d23[:, 3529] * args.l23_kappa
        merged[down23_key] = d23
        print(f"Applied symmetric x{args.l23_kappa} calibration to Layer 23 graft neurons 1998 & 3529")

    save_file(merged, args.output)
    print(f"Saved candidate checkpoint to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
