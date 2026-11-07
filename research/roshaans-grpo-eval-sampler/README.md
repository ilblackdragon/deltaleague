# Continued GRPO at the evaluation sampler: an evaluation-shaped mix, a 6k training cap and sampler-matched log-probs

Research by roshaans with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets, public research PRs and the `delta` CLI. No private test material.

- **Parent:** `9492e219-6f07-4bdc-b863-e86274ee62c2` (ma4ypic4y, "GRPO at the evaluation sampler, 20 steps")
- **Evaluation configuration:** `public-ladder500-8k-128-v1`
- **Submission:** `96257276-3a09-46a6-b095-c39e58b166f2`

All scores below are local, advisory estimates unless marked verified.

## Hypothesis

On-policy GRPO under the exact evaluation sampler, with a reward identical to strict scoring, improves the private score directly. Continuing it naively at the 8,192-token evaluation cap lengthens answers until truncation erases the gain. A training mix shaped like the evaluation, a training cap below the evaluation cap, and log-probs computed under the sampler should make further steps pay off.

## Method

- **Sampler:** thinking on, temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5. This matches `evaluation.json`.
- **Reward:** 1 if the answer is correct, `</think>` is closed and the output isn't truncated; otherwise 0. Correctness uses `delta_cli.scoring.score_answer` (`math`, `choice` and `ifeval` scorers).
- **Objective:** group-mean advantage with no std normalisation (Dr. GRPO). Token log-prob sums are weighted by advantage and divided by a constant. One optimizer step per rollout batch, fully on-policy. Groups whose rollouts all scored the same are skipped.
- **Optimizer:** AdamW, lr 1e-6, betas (0.9, 0.99), no weight decay, gradient clip 1.0. Master weights are fp32 with bf16 autocast. Checkpoint tensors the HF model doesn't load (MTP head) are copied through unchanged, and stored dtypes are preserved.
- **Batch:** 64 prompts × 8 samples per step.
- **Training mix:**
  - 35% math: gsm8k and MATH train.
  - 30% multiple choice: ARC-Challenge, CommonsenseQA, OpenBookQA, WinoGrande, HellaSwag and BoolQ train, plus the MMLU auxiliary train set.
  - 25% instruction following: `argilla/ifeval-like-data` (filtered), with exact matches to Google IFEval removed, scored by the IFEval rule checkers.
  - 10% logic: BIG-Bench train splits of tasks that are not in BBH, capped at 3,000 items per task.
- **Training cap:** rollouts stop at **6,144** tokens during training; evaluation stays at 8,192.
- **Sampler-matched log-probs:** each token's log-prob is computed under the evaluation sampler's distribution, not the raw model's: presence penalty from first occurrence in the completion, then top-k 20 and top-p 0.95, renormalised over a top-64 candidate pool. In our runs, 99.9% of sampled tokens fall inside the kept set.
- **Contamination guard:** the gsm8k and MATH train items used by our local probe are removed from training. No evaluation test splits or public validation items are used for training.

## Local evaluation

Every candidate is scored next to 9492e219 on the same machine, on identical questions, with strict scoring (unfinished answers count 0):

- **Math probe:** 512 held-out gsm8k and MATH train items with the evaluation sampler.
- **Benchmark set:** 975 items over bbh, gsm8k, ifeval, math500 and mmlu_pro, using the evaluation's lm-eval prompts.

## Results (paired against 9492e219)

| Candidate | Math probe | Benchmark set |
|---|---:|---:|
| **Submitted: mix + 6k cap + matched log-probs, 36 steps** | **+2.93 ± 1.86** | **+3.28 ± 1.67** |
| Same recipe, 24 steps | +3.91 ± 1.85 | −0.31 ± 1.57 |
| Mix + 6k cap, raw log-probs, 30 steps | +3.71 ± 1.94 | +1.54 ± 1.61 |
| Mix at the 8k cap, raw log-probs, 30 steps | +2.93 ± 1.92 | +0.92 ± 1.63 |
| Math + multiple choice at the 8k cap, 20 steps | −4.10 ± 1.90 | −4.31 ± 1.63 |
| Math + multiple choice at a 4k training cap, 20 steps | +0.39 ± 1.73 | −3.59 ± 1.67 |

The submitted checkpoint on the benchmark set, by family: gsm8k +7.8, mmlu_pro +5.7, ifeval +2.1, math500 +1.6, bbh −1.1. It finishes 84.6% of math-probe answers and 94.2% of benchmark answers; the parent finishes 87.5% and 95.3%.

## Negative findings

- **Continuing GRPO at the evaluation cap** lengthens answers every few steps (median 1,735 → 2,349 tokens over 20 steps), and the lost answers outweigh any gain.
- **A 4k training cap** stops the drift but makes the model under-think: gsm8k −10 on the benchmark set, from reasoning slips rather than formatting.
- **Extrapolating the 9492e219 delta** (θ + (α−1)Δ) hurts at every strength: probe −4.9 at α = 1.5, −12.5 at α = 2, −20.9 at α = 3. In bf16, the parent's delta is mostly single-step rounding moves, so small α isn't representable.
- **Updating only the top 20% highest-entropy tokens** avoids the length collapse of full-token updates on the same data (probe +1.2 vs −4.1 at step 20), but adds no gain on its own.
- **Averaging that run's delta with a gainful one** costs about 7 points on IFEval.

## Limitations

- The math probe uses the same `\boxed{}` prompt format as training; the benchmark set uses the evaluation's own prompts.
- The submitted checkpoint thinks somewhat longer than its parent (median 1,624 vs 1,297 tokens on the benchmark set) and truncates slightly more often on the math probe (79/512 vs 63/512).
- Paired local standard errors are about 1.6–1.9 points per test, and the private test has its own noise of about ±2.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --parent 9492e219-6f07-4bdc-b863-e86274ee62c2 --output dl_grpo
python grpo.py dl_grpo/model run --steps 24 --save-at 24 \
  --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 --train-max-tokens 6144 --match-sampler
python grpo.py run/step24 run2 --steps 12 --save-at 12 \
  --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 --train-max-tokens 6144 --match-sampler
delta patch build --parent dl_grpo/model --modified run2/step12/model.safetensors --output submission
delta validate submission
```

- The second command starts from the step-24 checkpoint with a fresh optimizer state.
- `setup_e14.sh` and `setup_e15.sh` are the exact pipelines that were run.
- `probe.py` and `gen_val.py` produce the two local tests.
- **Seeds:** prompt order `random.Random(1234)`; vLLM sampling seed `1000 + step`. GPU kernels are not bitwise deterministic, so a rerun reproduces the recipe, not the exact tensors.
- **Compute:** one A100 80GB, vLLM 0.30.0, `VLLM_ALLOW_INSECURE_SERIALIZATION=1` for in-place weight reloads.

**Expected sha256:**
- checkpoint `model.safetensors`: `ef02620762b89fbb4c76adf62313c448f4d89bab6a43a5b5986b01b4e2fd4ce0`
- patch `weights.safetensors` (sparse, 265 tensors): `967ce98fb54ac1c49e6fa6c42341dcf465b15f38ab5d1e931b79dee1677cbc2b`
- `manifest.json`: `8eebc9480566168a28f3be8e2aa43e0a3ad0226825151fe34b4c5db8f7a79967`

## Boundaries

- Only this experiment directory changes.
- No weights, credentials, private test material, signed URLs or raw generation logs are included.
