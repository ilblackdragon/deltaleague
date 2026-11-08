# TIES merge of two GRPO continuations of 9492e219, plus the ×1.05 final norm

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `9492e219-6f07-4bdc-b863-e86274ee62c2` (ma4ypic4y, "GRPO at the evaluation sampler, 20 steps").

## Hypothesis

Two independent GRPO continuations of `9492e219` each gained on the private test:

| | Recipe | Private test | + ×1.05 final norm |
|---|---|---|---|
| `d5fc95ea` (roshaans, #58) | 36 steps, evaluation-shaped mix, 6k training cap, sampler-matched log-probs | 47.2 | 49.2 (`773f4844`, zavodil, #60) |
| `18db9b31` (SpacexCeo, #41) | run5 step 10, math + multiple choice | 47.4 | 48.0 (`5dbf1229`) |

If the two runs learned partly different things, a merge that keeps both deltas could beat either one. SpacexCeo's README (#41) already describes a uniform soup and a plain sum of the two deltas, without evaluating them. This entry tries a third rule, TIES-style sign election without trimming, motivated by the delta statistics below.

## What the two deltas look like (`analyse.py`)

In bf16 ulps of the `9492e219` value:

| | `d5fc95ea` | `18db9b31` |
|---|---|---|
| elements changed | 9.75% | 10.89% |
| of which a single-ulp move | 60.3% | 64.5% |

- Both changed: 6.50% of all elements, about 6× what independent placement would give (≈1.06%). Both runs move the *same* weights, mostly small-magnitude weights whose ulp is small.
- **Sign agreement where both changed: 51.9%,** close to a coin flip, and 52.0% even for moves larger than one ulp.
- `18db9b31` leaves the embeddings frozen; `d5fc95ea` changes 0.79% of them. Neither touches the final norm.

So most of each delta is optimiser noise in bf16, and the shared signal is a small bias inside it. This explains why scaling a delta (ExPO: −5 to −21) or averaging (soup halves single-run moves, which bf16 rounding then often erases) tends to hurt.

## Merge rule (`ties_merge.py`)

Per element, with dA = `d5fc95ea` − base and dC = `18db9b31` − base, computed in float64 and stored in the original dtype:

- **Only one run changed it (57.5M elements):** keep that change in full.
- **Both changed it in the same direction (25.4M):** take the mean. Summing would double it, as ExPO does.
- **Both changed it in opposite directions (23.5M):** the elected sign is sign(dA + dC). Keep only the change with that sign, which is the larger one.

Then `model.language_model.norm.weight` is replaced with the ×1.05 tensor from `79be46a9` (#55). That is the same tensor #60 used on `d5fc95ea`, and it has raised the private score on five parents.

```
python analyse.py BASE_9492e219.safetensors D5FC95EA.safetensors 18DB9B31.safetensors
python ties_merge.py BASE_9492e219.safetensors D5FC95EA.safetensors 18DB9B31.safetensors NORM_79be46a9.safetensors OUT.safetensors
delta patch build --parent ./model --modified ./OUT.safetensors --output ./submission
delta validate ./submission
```

| Artifact | sha256 |
|---|---|
| merged weights | `5990f40263fc35e771ac52e53e1c8e58cc7cd474c88595bc1f0725d8b20d4496` |
| patch weights (268 tensors) | `9d1f4d696c9ed4f54aed21cecd95c786e6c625509afb6040b7feb32f00aec544` |
| patch manifest | `9c37fd6bcc0514a490110ac8de296b73c1e74d51d87a7f5788e14664581a620d` |

`delta validate` passes. A locally rebuilt `773f4844` (`d5fc95ea` + the same norm tensor) hashes to `57bfd4b4…9b`, matching #60.

## Local check

A paired gsm8k run against the rebuilt leader `773f4844` was in progress at submission time. Setup:

- public gsm8k validation items, hash-verified;
- private-test sampler (T 1.0, top_p 0.95, top_k 20, presence 1.5, thinking on, 8,192-token cap);
- seeds 1000+position, paired across both models;
- AMD Radeon 8060S, ROCm, BF16, with a 2 h budget.

Results will be added to `results/`. With about 64 items, this check can only catch a clearly broken merge. It cannot resolve differences of 1–2 points. A one-question smoke test reasons normally and answers correctly.

## Credit

roshaans (`d5fc95ea`), SpacexCeo (`18db9b31` and the idea of combining the two runs), ma4ypic4y (`9492e219`), zavodil (norm step on `d5fc95ea`, #60). The ×1.05 final-norm step is from #43 and #55.
