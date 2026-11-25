# Research trace (chronological, kept as the work happens)

Attribution: proposed and run by an AI agent (Claude Code, Opus 5.5), directed, budgeted and approved by ALuhning.
Only public material was used: the parent checkpoint, `evaluation.json`, `validation.json` (public), public
leaderboard metadata and research PRs, and the gsm8k / MATH *training* splits. No private test data.

Parent `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` (content hash `708085f8…37b8`, model.safetensors sha256
`04b1c301…4696`). Evaluation config `private1000-v6-thinking8k-20261111` (protocol `delta-public-v1`).
Original on the board: validation 40.29, private test 17.40.

## 1. Rejected first idea: final-norm temperature x1.4

Edit `w' = 1.4·(1 + w) − 1` on `model.language_model.norm.weight` (exact logit x1.4). Before running anything we
found it is not new: #7 (x1.25 and x1.667) and #42 (x1.25, rejected as duplicate content) made the same edit;
private test 19.0–20.5 (x1.25) and 14.7–17.2 (x1.667), and truncation *rose* with the gain (74.6 → 82.6%).
x1.4 would only interpolate a flat-to-negative curve, so it was dropped without spending GPU time or quota.

## 2. Mechanism no one had touched: linear-attention memory/erase gates

Searching all PR titles and bodies (107) and all 94 leaderboard names for A_log, dt_bias, decay, forget,
linear attention, DeltaNet, conv1d, in_proj_a/b: no hits. Qwen3.5-0.8B has 18 Gated-DeltaNet layers whose
per-token state decay is `exp(-exp(A_log) · softplus(in_proj_a(x) + dt_bias))`.

Measurements (`gate_probe.py`, `erase_triggers.py`, CPU, a 403-token two-question trace written for the probe):
- 108 of 288 linear-attention heads are long-memory (median half-life ≥ 256 tokens); on ordinary tokens many
  remember for 10^3–10^4+ tokens (the a=0 estimate of 400–2,300 tokens was wrong).
- A built-in erase signal fires at boundaries: right after `</think>` (84–93 of 108 heads at once),
  `<|im_end|>` / `<|im_start|>` (75–84), the start of a new question (62), and line / paragraph breaks.
  At those tokens half-life drops to 7–190 tokens.
- A hand-set A_log x0.5 and a per-head dt_bias shift both halve the eraser too: no one-knob edit extends
  memory while keeping the eraser intact.

## 3. Red-team and pivot: gate-only GRPO

Hand-set gate values are guesses with unknown sign; closed-form single edits on this parent have moved the
private test 0–8 points (SE ≈ 1.6). Large gains on this model came from on-policy RL at the evaluation sampler.
Chosen experiment: GRPO that trains only the gates (72 tensors, 590,400 parameters), reward = correct answer
with closed thinking within the cap. Gated by a failure-mode check first.

## 4. Failure-mode check (GPU, aitterminator: Strix Halo / Radeon 8060S, vLLM 0.33.0+strix ROCm image)

Original model, 128 held-out *training-split* problems (64 gsm8k train, 64 MATH train L1–4; `prep_data.py`, seed
20261124) x 2 samples, private-test sampler from `evaluation.json` (thinking on, T 1.0, top_p 0.95, top_k 20,
presence 1.5, 8,192 tokens), seeds 20261124 + 1000·item + sample. 2.04M tokens in 41.7 min (vLLM ~500–1,300 tok/s).
Note: served with the local ROCm vLLM image, not the platform's pinned CUDA image; same sampler settings.

| class | share | correct | mean tokens | restarts (Wait/Hmm/Alternatively/verify) | repeated 30-word windows | erase events / head / 1k tok |
|---|---|---|---|---|---|---|
| finished | 7% (18) | 100% | 5,082 | 16.9 (3.3 per 1k) | 0.0 | 18.9 (n=12) |
| never closed thinking | 91% (234) | 0% | 8,192 | 55.9 (6.8 per 1k) | 1.9 | 14.9 (n=12) |
| closed, answer cut off | 2% (4) | 0% | 8,192 | 33.8 | 0.0 | 17.4 (n=4) |

Reading: the dominant failure is endless re-verification with fresh wording (presence penalty), not verbatim
loops and not answer loops. Looping traces show fewer erase events than finished ones (confounded: finished
traces contain the answer section and its boundary tokens; small n). This rejects "paragraph breaks wipe
the notes" and does not establish a direction for hand-set gate changes.

Adjustments before training: practice cap 6,144; +0.25 reward for closing thinking and stopping (even if wrong);
held-out evals at steps 3/6/8 with the probe's sample-0 seeds (baseline truncation 93.75%); kill rule: stop after
step 3 if held-out truncation > 88.75%.

## 5. Hand-set arm: memory doubling (A_log += ln 0.5 on 108 long-memory heads)

`make_memory_edit.py model-orig sample_trace.txt model-mem` (only `*.linear_attn.A_log`, float32;
model.safetensors sha256 `92c6a130…1d65`). Same 128 held-out problems, sample-0 seeds, 8,192 tokens:

| | finished | never closed | answer cut off | correct |
|---|---|---|---|---|
| original | 8 (6.3%) | 118 | 2 | 6.3% |
| memory x2 | 7 (5.5%) | 119 | 2 | 5.5% |

Null result: doubling long-head memory does not change re-verification or accuracy. Not submitted.

## 6. Gate-only GRPO, run 1 (stopped after step 0: no learning signal)

Config: 32 prompts x 8 samples from gsm8k+MATH train, practice cap 6,144, lr 3e-4 (Adam, fp32 master), clip 1.0.
Step 0: 256 rollouts in 27.3 min; correct 2.0%, closed 2.3%, truncated 98.0%; **2 of 32 groups** had any reward
variance; update 4 min, grad norm 0.035. At that rate eight steps would carry ~16 informative groups: stopped.

## 7. Gate-only GRPO, run 2 (`runs/gates2`)

Changed only what the sparse signal forced: practice on gsm8k train only (parent finishes ~9% within 8K vs ~3%
for MATH), 16 prompts x 16 samples (same 256 rollouts/step; P(group has a success) ~0.78 vs ~0.2), practice cap
8,192 (= test cap). Restarted from the parent. Evals at steps 3 and 6; kill rule unchanged (truncation > 88.75% at step 3).
Step 0: 256 rollouts in 43.7 min; correct 3.5%, closed 3.5%, truncated 96.5%, mean 8,114 tokens; **2 of 16 groups**
informative; update 13.9 min, grad norm 0.024. Stopped during step 1 rollouts (≈1 h/step; six steps ≈ 6 h for ~12
informative groups). `policy-step1` (one update) is kept but not evaluated or submitted.

**Finding:** under the private sampler the parent closes thinking within 8,192 tokens on ~3–7% of problems, so
on-policy RL from the original model gets almost no reward signal at this hardware's throughput (~600–1,500 tok/s).
Gate-only RL is untested, not refuted: it needs a parent that already finishes often, or far more compute.
GPU use: pipeline paused 19:27–22:25 (≈3 h) on aitterminator; no rented compute; no submission.

## 8. Overnight run (original-model lineage only), `overnight.sh`

Decision (ALuhning): keep the lineage on the original; spend what it takes on aitterminator overnight.
- Training speed: fla 0.6.0 (`flash-linear-attention` + `fla-core`, Triton on ROCm) replaces the PyTorch
  reference gated-delta kernel in the trainer: 4.0 s vs 11.8 s per 4,096-token forward+backward; hidden-state
  cosine 0.99982, gate-gradient cosine ≥ 0.9936, grad-norm ratio 1.005 vs the reference.
- Filter: original model, first 1,024 gsm8k-train problems of the seeded shuffle x 2 samples, cap 6,144, seed
  20261125; pool = problems solved at least once (`make_pool.py`).
- GRPO run 3 (`runs/gates3`): pool, 12 prompts x 16, cap 8,192, lr 3e-4, 6 steps, evals at 3/6 (sample-0 seeds),
  kill rule truncation > 88.75% at step 3. Trains all positive-advantage samples plus up to max(4, #positives)
  failures per group, importance-weighted (unbiased group gradient).
- Public suite (`delta public-eval run`, pinned manifest, 1,110 items) for original and trained, then
  `delta compare`. Served with the ROCm vLLM image; the trained checkpoint is labelled by its file sha256 (local
  label, not the platform content hash).
Filter: 2,048 generations, 6.2% success at cap 6,144; 99 of 1,024 problems solved at least once (pool).
Run 3 steps 0–2 (practice pool): correct 50.0 / 55.7 / 52.6%, closed 52.1 / 58.3 / 54.7%, 11–12 of 12 groups
informative; ≈45 min per step. Held-out eval of `policy-step3`: truncation 93.75% (original 93.75%), finished
8 vs 8 of 128 (paired diff +0.000, 95% CI [−0.055, +0.055], 6 wins / 6 losses), correct 5.5% vs 6.3%.
Kill rule fired 04:55:32; public suite skipped; rfp-pipeline resumed 04:55:36 by the script's exit trap.
Outcome: negative; nothing submitted.

## 9. Next hypothesis from the data: the model finds the answer but does not commit

Text check on held-out cut-off gsm8k traces: the gold number appears in the thinking of 75% (median 15% into
the trace), but that loose match includes list numbers and hypotheticals; a strict "= answer" pattern found 3%
(it also missed 2/3 of finished correct traces, so it is too strict). Undecided from text, so measured directly.

**Forced-exit test** (`forced_exit.py`, GPU to itself, 07:00–07:17): the 118 sample-0 held-out traces cut off
inside thinking, cut at 25/50/75% of their length (previous line break), closed with `</think>`, answer sampled
(T 1.0, top_p 0.95, top_k 20, 384 tokens, seed 1, no presence penalty in HF generate). These traces scored 0%.

| | 25% | 50% | 75% | any cut |
|---|---|---|---|---|
| gsm8k (59) | 27.1% | 28.8% | 27.1% | 42.4% |
| MATH (59) | 11.9% | 13.6% | 8.5% | 18.6% |
| combined (118) | 19.5% | 21.2% | 17.8% | 30.5% |

Some forced exits kept reasoning after `</think>` and ran out of the 384-token answer budget, so this is a lower
bound on what the model "holds". Pre-set bar was ">= 25% at some cut": met on gsm8k, not on the combined set.

Incident: an earlier attempt (06:15) ran beside Command A+ with ~4 GB free host memory; the kernel OOM killer
killed llama-server, llama-swap restarted and stuck stopping Qwen3.6; fixed with the user's approval
(`podman stop qwen3.6-35b-a3b`), Command A+ ready 06:33. Lesson: GPU work only with the pipeline paused and
models unloaded.

## 10. Consistency-gated commit LoRA (`run_commit.sh`, 11:14–)

Data (training problems only; gold answers used only to select examples):
- Commit (`commit_data.py`): 1,200 cut-off gsm8k-train attempts from the filter (cap 6,144), forced exits at
  20/35/50/65/80% via vLLM completions (private sampler, 384-token answer, seed 20261126 + 10·trace + cut).
  304 traces had a correct forced exit somewhere; **198** had two cuts agreeing on the correct boxed answer ->
  example = prompt + thinking to the later agreeing cut + `</think>` + that forced answer. Plus **126** naturally
  finished correct attempts.
- Repair (`hint_data.py`): 800 MATH-train problems with the first half of the reference solution as a hint
  (no `\boxed`/answer), cap 4,096. **0 kept**: 797/800 still ran past the cap; 461 mentioned the hint. Lesson:
  the original cannot produce repair data; generate it with a model that already commits (STaR round 2).

LoRA (`train_lora_commit.py`): r 16, alpha 32, q/k/v/o + in_proj_qkv/in_proj_z/out_proj + gate/up/down in all
24 layers (150 targets, 10.2M params), lr 1e-4 cosine, 21 steps of 16, completion-only loss, seed 20261126;
loss 0.72 -> 0.68 (flat). Exported as a Delta `lora` patch (fp32 A/B, alpha 32; 40.9 MB), `delta validate` OK;
evaluated checkpoint rebuilt with `delta_cli.artifacts.reconstruct` (`reconstruct_eval.py`).

Held-out (128 problems, sample-0 seeds, private sampler, 8,192), `heldout_compare.py`:

| | correct | closed | truncated | mean tokens | paired diff (95% CI) | wins/losses |
|---|---|---|---|---|---|---|
| gsm8k | 5 -> 29 | 5 -> 35 | 59 -> 35 | 8,020 -> 6,627 | +0.375 [+0.266, +0.500] | 24/0 |
| MATH | 3 -> 14 | 5 -> 16 | 61 -> 50 | 7,967 -> 7,203 | +0.172 [+0.078, +0.281] | 12/1 |
| all | 8 -> 43 | 10 -> 51 | 120 -> 85 | 7,994 -> 6,915 | +0.273 [+0.195, +0.352] | 36/1 |

No closed-and-completed answer was wrong. MATH improved without any MATH training data. Kill rule (>= 15/128) passed.
Caveat: held-out problems are from the same training-split distributions as the commit data (not public
validation); one sample per problem.

## 11. Public suite (pinned `delta public-eval run`, ROCm vLLM, config `private1000-v6-thinking8k-20261111`)

With the evaluation config, public validation runs at the private sampler: thinking on, T 1.0, top_p 0.95, top_k 20,
presence 1.5, 8,192 tokens (`generation.public_validation`). First attempt failed on arguments (`--manifest` is required
with `--evaluation-config`). Second attempt: **both** models completed the same 8 benchmarks (502/1,110 items) and
then failed identically entering math500 with an `AssertionError` (task loading and the gold-solution helpers pass
on CPU, so it occurs during generation/scoring on this runtime; not investigated further: it affects the original
equally and the platform's evaluation is authoritative). No `result.json`, so `delta compare` could not run.

Paired local scores on the 8 completed benchmarks (single sample each; ±~6 points per benchmark at n = 64):

| benchmark | n | original | candidate | diff |
|---|---|---|---|---|
| bbh | 54 | 7.4 | 27.8 | +20.4 |
| gsm8k (flexible) | 64 | 43.8 | 53.1 | +9.4 |
| commonsense_qa | 64 | 54.7 | 57.8 | +3.1 |
| arc_challenge | 64 | 31.2 | 32.8 | +1.6 |
| boolq | 64 | 65.6 | 65.6 | 0.0 |
| arc_easy | 64 | 64.1 | 62.5 | −1.6 |
| hellaswag | 64 | 46.9 | 45.3 | −1.6 |
| ifeval (prompt strict) | 64 | 68.8 | 59.4 | −9.4 |
| mean of 8 | | 47.80 | 50.54 | +2.74 |

The ifeval drop (6 of 64 prompts) is a possible regression in instruction following and is reported as such.

## 12. Verified result

Submitted once with `--pr` #109 (submission `b94b64fe-1aac-4394-b7fa-6a7be1d3f8b0`, version `15049dff…`).
Platform, same config as the original: private test 17.40 ± 1.20 -> 20.40 ± 1.27 (+3.0, not separable by the
board's 1.96·√(se² + se²) ≈ 3.4 band); validation 40.29 -> 41.99 (+1.70); math500 +14.1, gsm8k +7.8,
bfcl single +6.9, ifeval +4.7, bbh −5.6, bfcl multi −3.1. Rank 91 -> 89 of 97. The local held-out estimate
(+27.3 on training-split maths) did not transfer at that size to the mixed private test.
