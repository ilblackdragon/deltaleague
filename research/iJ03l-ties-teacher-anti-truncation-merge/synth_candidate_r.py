import argparse
import gc
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def svd_compress_delta(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """Project a 2D task-vector delta onto its top-r singular subspace."""
    r = min(rank, delta.shape[0], delta.shape[1])
    if r <= 0:
        return torch.zeros_like(delta)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    recon = (U[:, :r] * S[:r]) @ Vh[:r, :]
    del U, S, Vh
    return recon


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate R: Rank 1 fb984a19 Backbone + Co-Fused Attention & MLP Teacher Task Vectors (Layers 0-18)")
    parser.add_argument("--anchor", default="/home/x-5194/NearRadar/experiment/model/model.safetensors", help="Rank 1 anchor (fb984a19)")
    parser.add_argument("--donor", default="/home/x-5194/NearRadar/models/0c834b41/model.safetensors", help="Validation and code donor (0c834b41)")
    parser.add_argument("--output", default="/home/x-5194/NearRadar/experiment/candidate_r.safetensors", help="Output weights")
    parser.add_argument("--beta-mlp", type=float, default=0.22, help="Strength of SVD task vector in MLP layers")
    parser.add_argument("--beta-attn", type=float, default=0.16, help="Strength of SVD task vector in linear attention layers")
    parser.add_argument("--rank-mlp", type=int, default=16, help="SVD rank for MLP task vector")
    parser.add_argument("--rank-attn", type=int, default=8, help="SVD rank for attention task vector")
    parser.add_argument("--max-layer", type=int, default=18, help="Max layer to apply task vectors (Layers 0-18 only)")
    args = parser.parse_args()

    if not os.path.exists(args.anchor) or not os.path.exists(args.donor):
        print(f"Missing input models: anchor={os.path.exists(args.anchor)}, donor={os.path.exists(args.donor)}", file=sys.stderr)
        sys.exit(1)

    print(f"Synthesizing Candidate R (The Significant Breakthrough):")
    print(f"  Anchor: {args.anchor} (fb984a19, Rank 1)")
    print(f"  Donor:  {args.donor} (0c834b41, #1 Val, 50% code)")
    print(f"  Beta MLP: {args.beta_mlp} (rank {args.rank_mlp}), Beta Attn: {args.beta_attn} (rank {args.rank_attn})")
    print(f"  Target Layers: 0..{args.max_layer}")
    print(f"  Layers {args.max_layer + 1}..23 and norm.weight: 100% UNTOUCHED fb984a19 (preserving L23 graft & tool calling)")

    mlp_prefixes = set(f"model.language_model.layers.{i}.mlp." for i in range(args.max_layer + 1))
    attn_prefixes = set(f"model.language_model.layers.{i}.linear_attn." for i in range(args.max_layer + 1))

    merged = {}
    fused_mlp = 0
    fused_attn = 0

    with safe_open(args.anchor, framework="pt") as fa, safe_open(args.donor, framework="pt") as fd:
        keys = list(fa.keys())
        d_keys = set(fd.keys())

        for idx, k in enumerate(keys):
            ta = fa.get_tensor(k)

            is_mlp = any(k.startswith(p) and "weight" in k for p in mlp_prefixes)
            is_attn = any(k.startswith(p) and "weight" in k for p in attn_prefixes)

            if (is_mlp or is_attn) and k in d_keys and len(ta.shape) == 2:
                td = fd.get_tensor(k)
                delta = td.float() - ta.float()
                if delta.norm().item() > 1e-6:
                    rank = args.rank_mlp if is_mlp else args.rank_attn
                    beta = args.beta_mlp if is_mlp else args.beta_attn
                    comp = svd_compress_delta(delta, rank)
                    m = ta.float() + beta * comp
                    merged[k] = m.to(ta.dtype)
                    if is_mlp:
                        fused_mlp += 1
                    else:
                        fused_attn += 1
                    del comp, m, delta, td
                else:
                    merged[k] = ta
            else:
                # 100% untouched fb984a19 anchor (preserves tool calling, Layer 23 graft, and late layers)
                merged[k] = ta

            if (idx + 1) % 60 == 0:
                gc.collect()

    print(f"Fused {fused_mlp} MLP tensors and {fused_attn} Attention tensors in Layers 0..{args.max_layer}. Total tensors saved: {len(merged)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(merged, args.output)
    print(f"Candidate R saved to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
