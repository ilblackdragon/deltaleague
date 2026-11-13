# Unstick little Qwen: half-strength published L15/L19 repetition edit on top of b8af1dbc

Research by SpacexCeo (RAEN) with Claude Code agents (an orchestrator and seven research subagents). Done 2026-11-05.

## Hypothesis
The current #1 (`b8af1dbc`, xf3s "little Qwen v0.1", 36.0 private) removes the "doubt" direction from layers 0–13 on
top of `a9340e14`. Its own write-up notes the cost: without doubt the model **cannot climb out of verbatim grooves**, and the
published Sparks-of-Weight-Space-Programming edit (layers 15/19 MLP down_proj) lines up with a separate "stuck" direction
(cos 0.37/0.46) rather than with the doubt direction (cos 0.03). On `b8af1dbc` we measured that verbatim grooves and
answer-phase overflow, not re-check cycles, are now the main reasons for hitting 8,192 tokens. Hypothesis: adding the published
edit to `b8af1dbc` restores groove escape and cuts truncation, without undoing the doubt removal. At full strength (×1.0) it
also broke multiple-choice answering on a stress set (MMLU 50 → 27.5), so the submitted version uses **half strength (α = 0.5)**.

What the agents could access: public leaderboard rows, public checkpoints (`delta competition init --parent …`), public
datasets, the public research PR of xf3s, and a rented RTX 5090. No private test material.

## The edit
- Parent: `b8af1dbc-d52b-4bbc-a302-3505d897303c` (lineage `0ea62f40` → `a9340e14` → `b8af1dbc`).
- Evaluation config: `public-ladder500-8k-v2`.
- Change: the upstream sparse edit of `RikiyaT/lm_understand-logs@4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`
  (`exp8_math/runs/p50c/edit`) at half strength: W ← W + 0.5·(W_published − W) in float32, cast to BF16, for
  `layers.15.mlp.down_proj.weight` (64,904 entries change) and `layers.19.mlp.down_proj.weight` (64,448). `b8af1dbc` does not
  touch these tensors, so `rebuild.py` first verifies the upstream base and edited hashes, then interpolates.
- Nothing was fit to evaluation data. No embedding, norm or output row changes beyond what the parent already has.

## Reproduction
```sh
delta competition init qwen35-08b-v1 --parent b8af1dbc-d52b-4bbc-a302-3505d897303c --output exp && cd exp
U=https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit
mkdir -p upstream && curl -fL $U/manifest.json -o upstream/manifest.json && curl -fL $U/edit.npz -o upstream/edit.npz
python rebuild.py model/model.safetensors upstream edited.safetensors 0.5
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
```
Expected hashes: see `hashes.txt` (submitted α = 0.5: full checkpoint sha256 `7eb59466…c0ec`, patch `weights.safetensors`
sha256 `291996a8…5e60`, format sparse-diff, 2 tensors).

## Evidence (local, advisory — not official scores)
Harness: vLLM 0.30, bf16, the config's sampling (thinking on, T 1.0, top-k 20, top-p 0.95, presence 1.5, 8,192 tokens), on a
fixed **public** proxy of 320 prompts (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-Challenge 40; seeded sample, see
`proxy_build.py`). "strict" counts a row as correct only if it closed `</think>` after ≥ 50 thinking tokens and finished within
8,192 tokens; on 6 leaderboard checkpoints strict tracked private best (private ≈ 12.84 + 0.566·strict, r 0.94, LOO-RMSE 3.6).

| run | gens | acc | truncated | mean tokens | strict | math500 | gsm8k | mmlu_pro | arc_c |
|---|---|---|---|---|---|---|---|---|---|
| parent b8af1dbc, 4 samples | 1280 | 43.5 | 0.280 | 3548 | 43.0* | 33.1 | 65.9 | 21.9 | 73.1 |
| **α = 0.5 (submitted)**, seed sets 1+2, 3 samples each | 1920 | 46.0 | 0.178 | 2893 | 45.6 | 34.9 | 65.6 | 28.3 | 75.4 |
| α = 1.0, seed sets 1+2, 3 samples each | 1920 | 45.0 | 0.160 | 2716 | 44.3 | 37.5 | 62.9 | 25.6 | 70.0 |
*parent strict pooled with its other seed runs: 43.2 (5 samples/prompt). Parent per-bench columns above pool its 4-sample
run and its second-seed run (math500 33.2, gsm8k 66.0, mmlu_pro 22.8, arc_c 73.0).

- α = 0.5: truncation 0.28 → 0.18, mean output −18%; strict +2.3 (seed set 1) and +2.4 (seed set 2), pooled **+2.4 ± 1.2
  (z 1.9)** (+3.3 ± 1.2 after we fixed a reference-pooling bug; local prediction 37.7, 80% interval 35.6–39.8 — the verified
  33.0 fell below it); MMLU-Pro +5.5, ARC-C +2.4, MATH-500 +1.7, GSM8K −0.4. MC answers closed without a letter: 9.7% vs 12.4% (parent).
  Our pre-registered bar for a "certain" gain was +4.1 strict; this does not reach it, so we submit it as a test of the
  hypothesis, not as a guaranteed improvement. Calibrated local prediction: private ≈ 37.4 (80% interval 34.6–40.1).
- α = 1.0: strict +1.1 ± 1.3 and a multiple-choice regression on the stress set (below) — rejected.
- Accuracy among normally closed generations falls (60.8 → 55.7 at α = 0.5): more hard problems now close, some wrongly.
- Stress set (265 public prompts incl. IFEval, HumanEval, MBPP, BBH, MMLU + 5 fixed prompts): see `stress.md`.

## Negative findings from the same session (b8af1dbc stack, same harness)
- `</think>` row ×0.92 on top of the parent: +4.7 strict on one seed set, −5.3 on another (pooled −0.3); it converts
  answer-phase overflow into never-closed generations one for one. ×0.96: −3.3.
- Final-norm gain ×0.95 / ×0.9 (+close ×0.992): +1.9 (1 seed) / +0.7 ± 1.4 (3 samples): neutral.
- A think-start guard on the `</think>` row: −1.4 ± 1.8; a learned "premature close" direction on that row: −6.4 ± 2.1.
- Wait/Hmm row damping is redundant on b8af1dbc (the neuron-998 flip and the projection already remove them).
- Seed noise is large: the unchanged parent scored strict 37.8 and 44.1 on two seed sets of the same 320 prompts.
- The same published edit on the older leader `ab592312` scored 26.2 vs 28.2 private (another participant's entry); our
  claim is specific to the doubt-removed stack, where grooves are the dominant failure.

## Submission
Submission ID: `51721d34-391e-4c58-bc5d-57bccff5016e` (https://deltaleague.ai/submissions/51721d34-391e-4c58-bc5d-57bccff5016e/), submitted 2026-11-05 20:16 UTC, version `344c8376-66a1-413b-abca-da2b5b986044`.

**Verified (official) result: private test 33.0, validation 41.44, truncation rate 0.15.** Parent `b8af1dbc`: 36.0 / 41.57 / 0.23.
The hypothesis' mechanism held on the server (truncation 0.23 → 0.15), but the private score fell by 3.0 (server SE ≈ 2.0 per
entry, so about 1 SE of the difference): more generations finish, but fewer of the finished ones are right. This matches the
local warning we recorded (accuracy among normally closed generations 60.8 → 55.7) and argues that on this stack the private set
rewards answer quality per finished generation more than finishing itself. Validation by benchmark vs parent: gsm8k 48.4 vs 42.2,
math500 32.8 vs 29.7, ifeval 60.9 vs 57.8, mmlu_pro 32.9 vs 35.7, bbh 33.3 vs 40.7, arc_challenge 26.6 vs 29.7.

## Limitations
Local proxy is math-heavy and not the private mix; the improvement is a hypothesis with a measured mechanism (fewer
truncations), not a demonstrated score gain. α = 0.5 was chosen after seeing the α = 1.0 stress result (one extra comparison;
no further tuning). The official private score decides.
