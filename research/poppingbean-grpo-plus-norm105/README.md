# GRPO leader 9492e219 + our ×1.05 final-norm step

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `9492e219-6f07-4bdc-b863-e86274ee62c2` ("GRPO at the evaluation sampler, 20 steps", rank 1, private test 43.6).

## Hypothesis

`9492e219` is a full-parameter GRPO fine-tune of `344c8376`. A weight diff shows the following:

- 271 of 488 tensors changed: all MLP, attention and linear-attention projections, layer norms, and 40,883 embedding rows.
- The updates are small (about 0.16% relative Frobenius norm on the MLPs) but high-rank.
- **The final RMSNorm is bit-identical to `344c8376`'s.**

On `344c8376`, a ×1.05 gain on the final norm (#43, `cd0b95bb`) moved the private test from 35.0 to 37.2. It also lowered truncation from 17.8% to 16.2%. GRPO did not touch that tensor, so the same step can be stacked onto it unchanged.

The risk is that GRPO optimised the policy at the original logit scale and T = 1.0. A ×1.05 gain is the same as sampling at T ≈ 0.95, which could undo part of what GRPO learned.

Credit: the GRPO model is ma4ypic4y's work. The norm step comes from SpacexCeo's `27e49fd4` and #43. This entry only combines them.

## Reproduce

```
python make_edit.py GRPO_9492e219.safetensors PARENT_344c8376.safetensors X105_cd0b95bb.safetensors OUT.safetensors
delta patch build --parent ./model --modified ./OUT.safetensors --output ./submission
delta validate ./submission
```

`make_edit.py` asserts two things:

- `cd0b95bb` differs from `344c8376` only in `model.language_model.norm.weight`.
- The GRPO model's final norm equals `344c8376`'s.

It then copies `cd0b95bb`'s tensor into the GRPO model. The resulting gain is ×1.0499 on average, ranging from 1.047 to 1.053.

| Artifact | sha256 |
|---|---|
| edited weights | `69723d479a5e9797fc36da9274ff832bcb1a8532c00859bedc80d77b9ca49ec8` |
| patch weights (same tensor as #43) | `c12d3505fc9d846ba8582c3812ef752a4ed930b52916a8cd7d7eab5b80574c04` |
| patch manifest | `2b492c0aa0413e7f6a69580058032de5ad92ba975941ad50d1737820e566d46c` |

## Local check

Setup:

- Public gsm8k items in index order, hash-verified.
- Private-test sampling: T 1.0, top_p 0.95, top_k 20, presence 1.5, thinking on, 8,192-token cap.
- Seeds 1000+position, paired across the two models.
- AMD Radeon 8060S, ROCm, BF16.
- Rescored in-process with `rescore_gsm8k.py`, because the CLI's math scorer returns 0 on Windows (a math_verify multiprocessing timeout).
- The agreed 2 h budget stopped the run after **46 of 64 items**.

```
python gsm8k_sampled.py WORKSPACE GRPO_DIR EDIT_DIR REPORT_PREFIX GENERATIONS.jsonl --n 64 --max-tokens 8192 --deadline-minutes 105
python rescore_gsm8k.py WORKSPACE GENERATIONS.jsonl OUT
python compare_items.py OUT.base.json OUT.edit.json
```

| | GRPO 9492e219 | + ×1.05 |
|---|---|---|
| gsm8k accuracy (46 items) | 50.0 | 58.7 |
| mean tokens | 1,284 | 1,504 |
| truncated | 0% | 0% |

Paired difference: **+8.7 points**, 95% bootstrap CI [−6.5, +23.9], 9 wins and 5 losses, sign-test p = 0.42.

Reading:

- The direction is positive, but the result is not significant.
- Unlike the same step on `344c8376` (−19% tokens locally), reasoning got about 17% *longer* here, so any gain does not come from shorter traces.
- The private test SE is about 2.2 per entry, and the parent's 43.6 is an unusually large jump. Expect regression toward the mean.

Aggregate and per-item 0/1 results are in `results/`. No generations are included.

## Official result

Pending submission.
