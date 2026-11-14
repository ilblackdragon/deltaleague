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
    parser = argparse.ArgumentParser(description="Candidate Q: fb984a19 Anchor (Rank 1, 53.4%) + 1c8f7b6a/0c834b41 Code & IFEval SVD in Layers 0-19 MLP")
    parser.add_argument("--anchor", default="/home/x-5194/NearRadar/experiment/model/model.safetensors", help="Rank 1 anchor (fb984a19)")
    parser.add_argument("--donor", default="/home/x-5194/NearRadar/models/0c834b41/model.safetensors", help="Validation and code donor (0c834b41)")
    parser.add_argument("--output", default="/home/x-5194/NearRadar/experiment/candidate_q.safetensors", help="Output weights")
    parser.add_argument("--beta", type=float, default=0.25, help="Strength of teacher task vector in MLP layers")
    parser.add_argument("--rank", type=int, default=16, help="SVD rank for task vector")
    parser.add_argument("--max-layer", type=int, default=19, help="Max layer to apply task vector (0-19 MLP only)")
    parser.add_argument("--norm-gain", type=float, default=1.012, help="Gentle RMSNorm gain for length moderation")
    args = parser.parse_args()

    if not os.path.exists(args.anchor) or not os.path.exists(args.donor):
        print(f"Missing input models: anchor={os.path.exists(args.anchor)}, donor={os.path.exists(args.donor)}", file=sys.stderr)
        sys.exit(1)

    print(f"Synthesizing Candidate Q:")
    print(f"  Anchor: {args.anchor} (fb984a19, Rank 1)")
    print(f"  Donor:  {args.donor} (0c834b41, #1 Val, 50% code)")
    print(f"  Beta: {args.beta}, Rank: {args.rank}, Max Layer: {args.max_layer}")
    print(f"  Layers 20-23 and ALL attention layers: 100% UNTOUCHED fb984a19")

    target_layers = set(f"model.language_model.layers.{i}.mlp." for i in range(args.max_layer + 1))

    merged = {}
    fused_count = 0
    with safe_open(args.anchor, framework="pt") as fa, safe_open(args.donor, framework="pt") as fd:
        keys = list(fa.keys())
        d_keys = set(fd.keys())

        for idx, k in enumerate(keys):
            ta = fa.get_tensor(k)

            # Apply SVD delta ONLY to Layers 0-19 MLP down_proj, gate_proj, up_proj
            is_target_mlp = any(k.startswith(prefix) and "weight" in k for prefix in target_layers)

            if is_target_mlp and k in d_keys and len(ta.shape) == 2:
                td = fd.get_tensor(k)
                delta = td.float() - ta.float()
                if delta.norm().item() > 1e-6:
                    comp = svd_compress_delta(delta, args.rank)
                    m = ta.float() + args.beta * comp
                    merged[k] = m.to(ta.dtype)
                    fused_count += 1
                    del comp, m, delta, td
                else:
                    merged[k] = ta
            elif k == "model.language_model.norm.weight" and args.norm_gain != 1.0:
                merged[k] = (ta.float() * args.norm_gain).to(ta.dtype)
            else:
                # 100% fb984a19 anchor (preserves tool calling, Layer 23 graft, and all attention)
                merged[k] = ta

            if (idx + 1) % 60 == 0:
                gc.collect()

    print(f"Fused {fused_count} MLP tensors in Layers 0..{args.max_layer}. Total tensors saved: {len(merged)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(merged, args.output)
    print(f"Candidate Q saved to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
