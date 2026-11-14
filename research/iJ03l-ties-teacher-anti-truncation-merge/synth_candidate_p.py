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
    parser = argparse.ArgumentParser(description="Candidate P: Backbone 0c834b41 + Low-Rank Math SVD from fdad0374 in Layers 0-14 MLP")
    parser.add_argument("--backbone", default="/home/x-5194/NearRadar/models/0c834b41/model.safetensors", help="0c834b41 backbone")
    parser.add_argument("--math-donor", default="/home/x-5194/NearRadar/models/fdad0374/model.safetensors", help="fdad0374 math donor")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    parser.add_argument("--beta", type=float, default=0.20, help="SVD delta strength in layers 0-14 MLP")
    parser.add_argument("--rank", type=int, default=16, help="SVD rank")
    parser.add_argument("--max-layer", type=int, default=14, help="Max layer to apply math delta (inclusive)")
    parser.add_argument("--norm-gain", type=float, default=1.00, help="Final RMSNorm multiplier")
    parser.add_argument("--choice-gain", type=float, default=1.00, help="Choice tokens A-E gain")
    parser.add_argument("--think-end-gain", type=float, default=1.00, help="Closure token </think> gain")
    args = parser.parse_args()

    if not os.path.exists(args.backbone) or not os.path.exists(args.math_donor):
        print(f"Missing input models: backbone={os.path.exists(args.backbone)}, donor={os.path.exists(args.math_donor)}", file=sys.stderr)
        sys.exit(1)

    print(f"Synthesizing Candidate P:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Math Donor: {args.math_donor}")
    print(f"  Beta: {args.beta}, Rank: {args.rank}, Max Layer: {args.max_layer}")
    print(f"  Norm Gain: {args.norm_gain}, Choice Gain: {args.choice_gain}, Think End Gain: {args.think_end_gain}")

    target_layers = set(f"model.language_model.layers.{i}.mlp." for i in range(args.max_layer + 1))

    merged = {}
    fused_count = 0
    with safe_open(args.backbone, framework="pt") as fb, safe_open(args.math_donor, framework="pt") as fd:
        keys = list(fb.keys())
        d_keys = set(fd.keys())

        for idx, k in enumerate(keys):
            tb = fb.get_tensor(k)

            # Check if this tensor is in target Layers 0-14 MLP
            is_target_mlp = any(k.startswith(prefix) and "weight" in k for prefix in target_layers)

            if is_target_mlp and k in d_keys and len(tb.shape) == 2:
                td = fd.get_tensor(k)
                delta = td.float() - tb.float()
                if delta.norm().item() > 1e-6:
                    comp = svd_compress_delta(delta, args.rank)
                    m = tb.float() + args.beta * comp
                    merged[k] = m.to(tb.dtype)
                    fused_count += 1
                    del comp, m, delta, td
                else:
                    merged[k] = tb
            elif k == "model.language_model.norm.weight" and args.norm_gain != 1.0:
                merged[k] = (tb.float() * args.norm_gain).to(tb.dtype)
            elif k == "model.language_model.embed_tokens.weight" and (args.choice_gain != 1.0 or args.think_end_gain != 1.0):
                e = tb.clone()
                if args.choice_gain != 1.0:
                    for tid in list(range(32, 37)) + [2434, 3913]:
                        e[tid, :] = (e[tid, :].float() * args.choice_gain).to(tb.dtype)
                if args.think_end_gain != 1.0:
                    e[248069, :] = (e[248069, :].float() * args.think_end_gain).to(tb.dtype)
                merged[k] = e
            else:
                # 100% 0c834b41 backbone
                merged[k] = tb

            if (idx + 1) % 60 == 0:
                gc.collect()

    print(f"Fused {fused_count} MLP tensors in Layers 0..{args.max_layer}. Total tensors saved: {len(merged)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(merged, args.output)
    print(f"Candidate P saved to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
