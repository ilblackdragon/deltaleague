import os
import sys
import gc
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def svd_compress_delta(delta: torch.Tensor, rank: int = 16) -> torch.Tensor:
    """Computes truncated SVD rank compression on 2D weight delta tensor."""
    r = min(rank, delta.shape[0], delta.shape[1])
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    compressed = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return compressed

def main():
    base_path = "/home/x-5194/NearRadar/experiment/model/model.safetensors"
    policy_path = "/home/x-5194/NearRadar/models/ac61dc20/model.safetensors"
    out_path = "/home/x-5194/NearRadar/experiment/edited.safetensors"

    if not os.path.exists(base_path) or not os.path.exists(policy_path):
        print(f"Error: model files missing. base: {os.path.exists(base_path)}, policy: {os.path.exists(policy_path)}")
        sys.exit(1)

    print("Opening models via zero-copy safe_open...")
    beta_svd = 0.40  # Rank-16 anti-truncation policy injection factor
    alpha_1d = 0.25  # Smooth norm parameter interpolation

    merged = {}
    with safe_open(base_path, framework="pt") as f_base, safe_open(policy_path, framework="pt") as f_policy:
        keys = list(f_base.keys())
        policy_keys = set(f_policy.keys())
        total = len(keys)

        for idx, k in enumerate(keys):
            t_base = f_base.get_tensor(k)

            # Preserve token embeddings intact to prevent early input drift
            if "embed_tokens" in k:
                merged[k] = t_base
                continue

            if k in policy_keys:
                t_pol = f_policy.get_tensor(k)
                if len(t_base.shape) == 2 and min(t_base.shape) > 1:
                    # Truncated SVD Rank Compression on 2D weight matrices
                    delta = t_pol.float() - t_base.float()
                    compressed_delta = svd_compress_delta(delta, rank=16)
                    m = t_base.float() + beta_svd * compressed_delta
                    merged[k] = m.to(t_base.dtype)
                    del t_pol, delta, compressed_delta, m
                else:
                    # 1D norms / biases: smooth linear mode connectivity
                    m = (1.0 - alpha_1d) * t_base.float() + alpha_1d * t_pol.float()
                    merged[k] = m.to(t_base.dtype)
                    del t_pol, m
            else:
                merged[k] = t_base

            del t_base

            if (idx + 1) % 50 == 0 or idx == total - 1:
                print(f"Processed {idx + 1}/{total} tensors (SVD rank compression)...")
                gc.collect()

    # 1. Calibrated Final RMSNorm for optimal output logit temperature
    norm_key = "model.language_model.norm.weight"
    if norm_key in merged:
        merged[norm_key] = (merged[norm_key].float() * 1.040).to(merged[norm_key].dtype)
        print(f"✓ Calibrated {norm_key} by 1.040")

    # 2. Retain Active Phase-Gate Push in Layer 23
    down_key = "model.language_model.layers.23.mlp.down_proj.weight"
    if down_key in merged:
        down = merged[down_key].clone()
        down[:, 0] = down[:, 0] * 1.20
        down[:, 1] = down[:, 1] * 1.20
        merged[down_key] = down
        print(f"✓ Retained x1.20 phase-gate push on Layer 23 MLP")

    print(f"Saving SVD compressed checkpoint to {out_path}...")
    save_file(merged, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✓ Successfully saved {out_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
