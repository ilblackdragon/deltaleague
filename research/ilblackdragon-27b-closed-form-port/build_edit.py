"""Port the qwen35-08b "3-in-1" closed-form edit (a9340e14) to qwen38-27b-nvfp4-ref.

0.8B recipe (recovered by diffing a9340e14 against parent 0ea62f40; tied embeddings):
  * 7 Wait/Hmm token rows x0.5
  * </think> row x1.3
  * final RMSNorm gain (1 + w) x1.4  (Qwen3.5 RMSNorm is zero-centred)

27B differences handled here:
  * Embeddings are untied. The 0.8B edit scaled one shared row, i.e. both the input
    embedding and the output logit row. We scale both: embed_tokens (BF16) and lm_head.
  * lm_head is NVFP4 (packed E2M1 codes, FP8-E4M3 per-16-element block scales, F32
    global scale). A row is scaled by scaling its 320 FP8 block scales; the 4-bit codes
    are untouched. x0.5 is exact; other factors round per block to the nearest E4M3
    value. The realized row-norm factor is printed.
  * The final norm is BF16 and edited in float32, then cast.

Usage:
  python build_edit.py PARENT_MODEL_DIR OUT.safetensors --filler 0.5 --think 1.3 --norm 1.4
"""
import argparse
import hashlib
import json
import math

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

FILLER = ['Ġwait', 'wait', 'ĠWait', 'Wait', 'Hmm', 'Ġhmm', 'ĠHmm']
THINK_CLOSE = '</think>'
EMBED = 'model.language_model.embed_tokens.weight'
HEAD = 'lm_head.weight'
HEAD_SCALE = 'lm_head.weight_scale'
NORM = 'model.language_model.norm.weight'
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def block_sq_norms(packed_row):
    """Per-16-element block squared norms of one NVFP4 row (two codes per byte)."""
    lo, hi = packed_row & 0xF, packed_row >> 4
    vals = torch.stack([E2M1[lo.long()], E2M1[hi.long()]], -1).reshape(-1)
    return vals.reshape(-1, 16).pow(2).sum(-1)


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parent')
    ap.add_argument('out')
    ap.add_argument('--filler', type=float, required=True)
    ap.add_argument('--think', type=float, required=True)
    ap.add_argument('--norm', type=float, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.parent)
    filler_ids = tok.convert_tokens_to_ids(FILLER)
    think_id = tok.convert_tokens_to_ids(THINK_CLOSE)
    assert filler_ids == [3655, 11158, 13428, 13784, 77264, 84485, 85152], filler_ids
    assert think_id == 248069, think_id
    rows = {i: args.filler for i in filler_ids}
    rows[think_id] = args.think

    path = f'{args.parent}/model.safetensors'
    with safe_open(path, 'pt') as f:
        metadata = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    report = {'factors': vars(args), 'rows': {}}
    embed = tensors[EMBED]
    scale = tensors[HEAD_SCALE]
    packed = tensors[HEAD]
    assert embed.dtype == torch.bfloat16 and scale.dtype == torch.float8_e4m3fn and packed.dtype == torch.uint8
    embed, scale = embed.clone(), scale.clone()
    for i, a in rows.items():
        embed[i] = (embed[i].float() * a).to(torch.bfloat16)
        old = scale[i].float()
        new = (old * a).clamp(max=448.0).to(torch.float8_e4m3fn)
        scale[i] = new
        sq = block_sq_norms(packed[i])
        realized = math.sqrt(float((new.float() ** 2 * sq).sum()) / float((old ** 2 * sq).sum()))
        nz = old != 0
        per_block = new.float()[nz] / old[nz]
        report['rows'][tok.convert_ids_to_tokens(i)] = {
            'id': i, 'target': a, 'lm_head_row_norm_factor': round(realized, 5),
            'lm_head_block_factor_min': round(float(per_block.min()), 5),
            'lm_head_block_factor_max': round(float(per_block.max()), 5)}
    tensors[EMBED], tensors[HEAD_SCALE] = embed, scale

    norm = tensors[NORM]
    assert norm.dtype == torch.bfloat16
    tensors[NORM] = ((1.0 + norm.float()) * args.norm - 1.0).to(torch.bfloat16)
    realized = ((1.0 + tensors[NORM].float()) / (1.0 + norm.float()))
    report['final_norm_gain_realized_median'] = round(float(realized.median()), 5)

    for k in (EMBED, HEAD_SCALE, NORM):
        assert torch.isfinite(tensors[k].float()).all(), k
    report['changed_tensor_sha256'] = {k: tensor_sha(tensors[k]) for k in (EMBED, HEAD_SCALE, NORM)}
    save_file(tensors, args.out, metadata=metadata)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
