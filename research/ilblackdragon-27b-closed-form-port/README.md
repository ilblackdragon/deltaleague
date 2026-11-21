# Porting the 0.8B "3-in-1" closed-form edit to Qwen 3.8 · 27B NVFP4

Status: variant A scored **29.0 ± 2.6 private** (original 22.0) and **74.26 public validation** (original 73.34); it is rank 1 on the 27B board as of 2026-11-21. B and C are built and not submitted. No local accuracy evidence. This host has no GPU, and the NVFP4 checkpoint needs a Blackwell GPU to serve.

## Hypothesis

On `qwen35-08b-v1`, the first large private-test jump came from a training-free, three-part edit: `a9340e14` (0xZxh, "3-in-1 combo"). On the current 0.8B board (`private1000-v6-thinking8k-20261111`) it scores 30.5 private against 17.4 for the original, and truncation falls from 75.2% to 39.6%. It reduces long, unterminated thinking:

1. dampen the Wait/Hmm hesitation tokens,
2. raise the `</think>` logit,
3. sharpen sampling through the final RMSNorm gain.

The 27B competition's private test (`qwen38-27b-nvfp4-agentic300-8k-v1`) is a multi-turn tool-use track with a budget of **8,192 generated tokens per episode, thinking included**. Each turn's thinking is stripped, and a turn that does not close its thinking yields no action. Exhausting any budget scores 0. The original model scores 73.34 on public validation but only 22.0 on the private test. Hypothesis: much of that gap is budget exhaustion from long thinking, which the same edit should reduce at 27B. This is untested. A 27B may not share the 0.8B's failure mode, and sharper sampling may change JSON-protocol behaviour in either direction.

## What the 0.8B edit is

The recipe was recovered by diffing `a9340e14` against its parent `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` with `recover_08b_recipe.py`. Only two tensors differ.

- `embed_tokens` (tied with the output head), 8 rows changed:
  - ×0.5: `Ġwait` 3655, `wait` 11158, `ĠWait` 13428, `Wait` 13784, `Hmm` 77264, `Ġhmm` 84485, `ĠHmm` 85152;
  - ×1.3: `</think>` 248069.
- `norm.weight`: `(1 + w) · 1.4 − 1`. Qwen3.5 RMSNorm is zero-centred, so every logit is multiplied by 1.4, roughly temperature 0.71 under the configuration's T = 1.0.

## Port to 27B

The 27B parent is `47f586c3-5e3f-42ac-9991-e3db49e09103` (content hash `38df01e7…8e10`). Its architecture is `Qwen3_5ForConditionalGeneration`, with 64 layers and hidden size 5120. It uses the same tokenizer IDs, which `build_edit.py` asserts. Embeddings are **untied**.

- **Input embedding** `model.language_model.embed_tokens.weight` (BF16): the same row factors as 0.8B, computed in float32 and cast.
- **Output head** `lm_head` (NVFP4: packed E2M1 codes, FP8-E4M3 scale per 16 elements, F32 global scale): each row is scaled by multiplying its 320 FP8 block scales; the 4-bit codes are untouched.
  - ×0.5 is exact.
  - Other factors round per block to the nearest E4M3 value. The realized row-norm factor is reported, e.g. 1.292 for a 1.3 target.
- **Final norm** `model.language_model.norm.weight` (BF16): `(1 + w) · g − 1` in float32, then cast.

The 0.8B edit scaled one shared row, so the port scales both the input and output rows to keep the same semantics. Patches are sparse-diff replacements of 3 tensors (variant C: 1 tensor). No vision, MTP or quantized MLP/attention tensors change.

| Variant | Wait/Hmm rows | `</think>` row (realized lm_head) | Final-norm gain | Patch weights SHA-256 |
| --- | ---: | ---: | ---: | --- |
| A, faithful port | ×0.5 | ×1.3 (1.292) | ×1.4 | `d9c31be5…2213` |
| B, gentler | ×0.7 | ×1.15 (1.157) | ×1.2 | `962f3f9e…ad19` |
| C, norm-only ablation | — | — | ×1.2 | `aa165bb0…aba2` |

Full hashes and per-row realized factors are in `results/variant-*.json`. The doses for B and C were chosen before any evaluation; there was no search. The 0.8B history suggests the norm gain had diminishing and then negative returns once the model already stopped on time: ×1.2 hurt reasoning on a later leader (PR #83), while ×1.05 steps helped earlier parents (PR #60). B hedges against A being too strong for a larger model.

## Reproduction

Python 3.12, torch 2.15.0+cpu, transformers 5.22.1, safetensors 0.8.0, deltaleague CLI from `main` at `eded8b5`. CPU only; peak RSS about 24 GB; about 40 s per variant.

```sh
delta competition init qwen38-27b-nvfp4-ref --output ws && cd ws
python build_edit.py model A.safetensors --filler 0.5 --think 1.3 --norm 1.4
python build_edit.py model B.safetensors --filler 0.7 --think 1.15 --norm 1.2
python build_edit.py model C.safetensors --filler 1.0 --think 1.0 --norm 1.2
delta patch build --parent model --modified A.safetensors --output sub-A   # likewise B, C
delta validate sub-A
python verify_reconstruction.py model sub-A A.safetensors                  # bitwise check
```

The build is deterministic with no seeds; patch hashes should match the table.

## Verification done

- `delta validate` passed for all three.
- `delta_cli.artifacts.reconstruct` from parent + patch matches the edited checkpoint bit for bit across all 2,194 tensors, with 0 mismatches.
- All edited tensors are finite.
- The tokenizer IDs match the 0.8B IDs.

## Not done / limitations

- **No inference has been run on any candidate.** NVFP4 kernels need a Blackwell GPU, and none was available. There are no local accuracy, truncation or protocol-validity estimates; the platform's verified scores will be the first measurement.
- The `</think>` ×1.3 push scales a logit and does not add to it. Where that logit is negative, scaling it pushes the token further down. That held on 0.8B too, and the empirical result there was positive, but nothing here verifies the 27B sign pattern.
- The final-norm gain also changes the presence penalty's effective size relative to the logits, which divides it by the gain.
- 0.8B gains after this edit came mainly from GRPO at the evaluation sampler and from distillation. Those are not attempted here: 27B NVFP4 training is outside this budget.

## Submission

Variant A, second attempt: submission `27f09fa9-31fd-494a-a45e-8f169794969d`, submitted 2026-11-21 09:16 UTC after platform PR ilblackdragon/deltaleague-platform#33 was deployed. It passed verification in about 7 minutes and completed evaluation at 10:47 UTC.

| | Private test (300 agentic episodes, 8K budget) | Public validation |
| --- | ---: | ---: |
| Original model | 22.00 | 73.34 |
| Variant A | **29.00 ± 2.62** (87/300 passed) | **74.26** |
| Difference | +7.00 | +0.92 |

- Using only the two standard errors, the private gain is about 2.0 σ. The board's band treats rows as independent; a paired test on the shared 300 items would likely be tighter, but per-item results are not exposed.
- The platform's reported `truncation_rate` is 0.0 with `output_tokens` 0. Those fields look unpopulated for this track, so the result does **not** confirm that the gain came from fewer budget-exhausted episodes. That mechanism remains a hypothesis.
- Public validation changed little overall (+0.92); the per-benchmark scores are in the platform result. The sharper sampling did not visibly hurt the knowledge or coding benchmarks.

Variant A, first attempt: submission `d1f5f3eb-7124-4508-a7f6-2acbb9762bfd`, **rejected (`artifact_invalid`)**; no score. The cause is below.

The platform rejected the submission 6 s after upload. Verification currently cannot handle any patch on this 21.9 GB parent: the CPU verifier downloads the parent under the default 4 GiB upload size limit (`artifact_size`). Its process limits (8 GiB file size, 16 GiB address space) are also below what a full 27B reconstruction needs. At the time this was a platform limitation, not a defect of this patch; any 27B weight submission is affected, including variant C.

A second, independent blocker: the competition accepts edits only to tensors under `model.language_model.`. `delta_cli.artifacts.reconstruct(..., allowed_prefix="model.language_model.")` raises `frozen_tensor` for A, because A changes `lm_head.weight_scale`; it accepts C. At 0.8B the output head is tied and covered by that prefix. At 27B the head is untied, so the output-logit half of the recipe (Wait/Hmm ×0.5, `</think>` ×1.3 on logit rows) is not currently admissible, and neither is variant B. Plain `delta validate` does not apply the prefix, so participants cannot see this before upload.
