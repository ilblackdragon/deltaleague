# Self-distilled LoRA on 344c8376: rejection-sampled short correct traces (public train splits)

Research by SpacexCeo (RAEN) with Claude Code agents. Done 2026-11-06.

## Hypothesis
On the doubt-removed stack, head-level edits (`</think>` row, final-norm gain, token-row damping, groove-escape) only change
*when* the model stops, and the server rewards accuracy of finished answers (our earlier entry 344c8376 raised finishing but
lowered accuracy among finished answers). Rejection-sampling fine-tuning on the model's OWN correct, shortest, normally closed
traces should raise accuracy of finished answers and finishing together, without teaching new content.

What the agents could access: public leaderboard rows and checkpoints, public datasets (TRAIN splits only), rented GPUs.
No private test material; validation/test items were not used for training.

## Method
- Parent: `344c8376-66a1-413b-abca-da2b5b986044` (b8af1dbc + half-strength published L15/L19 edit; lineage
  `0ea62f40` → `a9340e14` → `b8af1dbc` → `344c8376`). Config `public-ladder500-8k-128-v1`.
- Data (`gen_rft.py`): 4,400 public TRAIN prompts (seed 20261106) — GSM8K train 1,400; MATH (EleutherAI/hendrycks_math, train,
  levels 1–4) 1,600; ARC-Challenge 500 + ARC-Easy 300 train; MMLU auxiliary_train 600 — 4 samples each (two passes of 2, seed
  offsets 0 and 500), plus MBPP non-test splits (train/validation/prompt, 474 prompts, 4 samples, scored by running the tests;
  `PSET=v3`). All from the parent with the competition sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 6,144-token
  cap). 19,496 generations in total. (An IFEval-like subset was planned but its prompts were skipped by the builder; none used.)
- Selection (`select_rft.py`): per prompt, among samples that are correct (math-verify / letter match), closed `</think>` after
  ≥ 50 thinking tokens and not truncated, keep the shortest → 3,723 traces (GSM8K 1,172, MATH 1,180, ARC 739,
  MMLU-aux 507, MBPP 125), mean 971 tokens.
- Training (`train_lora.py`): LoRA r 16, alpha 32, on all 150 language-model projections (attention q/k/v/o, linear-attention
  in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down), lr 5e-5 cosine with 10-step warm-up, 1 epoch, ~64k tokens per step (63 steps),
  loss on completion tokens only (chunked cross-entropy, verified equal to the model's own loss). Final train loss ≈ 0.31.
- Export (`export_lora.py`): Delta `lora` format (A [r,in], B [out,r] in float32, alpha 32), reconstructed with the platform's
  own `reconstruct()` for local evaluation. No embedding, norm, visual or MTP tensors change.

## Reproduction
`gen_rft.py PARENT_DIR gen.jsonl 2 6144 {0,1} 2` (two shards) → `select_rft.py train.jsonl gen_s0.jsonl gen_s1.jsonl --per 1
--max_tokens 6000` (plus `SEED_OFFSET=500` pass and `PSET=v3 ... 4` for MBPP) → `train_lora.py PARENT_DIR train.jsonl lora --r 16 --alpha 32 --lr 5e-5 --epochs 1
--tok_per_step 65536`
→ `export_lora.py PARENT_DIR lora sub model 344c8376-66a1-413b-abca-da2b5b986044 <parent content hash>`.
Sampling is seeded but vLLM generation is not bit-reproducible across hardware/batching; the submitted adapter's hash is in
`hashes.txt`. Hardware: 1× RTX 5070 Ti (train ~15 min).

## Evidence (local, advisory — not official scores)
Public proxy of 320 prompts (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-C 40; test splits), competition sampler, 3–4 samples per
prompt. "strict" = correct AND closed after ≥ 50 thinking tokens AND not truncated.

| model | gens | strict | acc | truncated | acc among normally closed | GSM8K | MATH-500 | MMLU-Pro | ARC-C |
|---|---|---|---|---|---|---|---|---|---|
| b8af1dbc | 1280 | 41.8 | 42.7 | 0.288 | 59.4 | 61.9 | 34.4 | 21.2 | 72.5 |
| parent 344c8376 | 1280 | 44.2 | 44.7 | 0.180 | 54.5 | 61.2 | 36.2 | 28.4 | 69.4 |
| **submitted LoRA (v3), seed sets 1+2** | 1920 | **47.5** | 47.7 | 0.14 | 55.2 | see below | | | |

- Seed set 1: strict 47.8 vs parent 44.7 (+3.1 ± 1.3); fresh seed set 2: 47.2 vs parent 45.8 (+1.35 ± 1.6).
  Pooled: **+2.5 ± 1.0 (z 2.5) vs the parent** (parent pooled over 10 samples/prompt), **+5.25 ± 1.16 (z 4.5) vs b8af1dbc**.
- Accuracy among normally closed generations is unchanged vs the parent (55.2 vs 55.2): the gain comes with more finished answers
  at the same accuracy, unlike the head-level edits we tried (they traded accuracy for finishing).
- Stress set (265 public prompts, 3 samples each; GSM8K/MATH/MMLU/BBH/IFEval 40 each, HumanEval/MBPP 30 each):

| bench | parent 344c8376 | submitted LoRA |
|---|---|---|
| all | 41.1 | 43.1 |
| IFEval | 57.5 | 62.5 |
| BBH | 28.3 | 34.2 |
| MMLU | 43.3 | 45.8 |
| HumanEval | 32.2 | 34.4 |
| MBPP | 32.2 | 33.3 |
| MATH-500 | 35.8 | 35.8 |
| GSM8K | 59.2 | 56.7 |

- Earlier variants (negative/neutral, not submitted): v1 (lr 1e-4, no MBPP, 2 samples/prompt) +3.0 then +0.3 on the second seed
  set (pooled +1.6) and lost HumanEval/MBPP on the stress set; v2 (r 32, lr 2e-4) −1.35 vs v1. Lower lr, 4 samples per prompt and
  MBPP traces fixed the code regression.

## Negative findings (same session)
Digit/letter row sharpening (s 1.15: +0.1, s 1.30: −7.8 strict); final-norm gain ×1.10 (−1.8); doubt projection extended to layers
14–15 (−0.8); published edit on layer 15 only (+2.6 vs b8af1dbc, same mechanism as the parent). See also research PR #24.

## Submission
Submission ID: `7123caf3-b7b1-4b53-bf5b-5e1e4f14e2d4` (https://deltaleague.ai/submissions/7123caf3-b7b1-4b53-bf5b-5e1e4f14e2d4/), submitted 2026-11-06, version `6c2acb28-4b90-42ad-8e12-55fdae047aff`.

**Verified (official) result: private test 32.4, validation 42.13, truncation rate 0.126.** Parent 344c8376: 35.0 / 41.09 / 0.178
(a single, likely lucky draw; its posterior mean across the two configurations is ≈ 33.7 ± 0.9, per-evaluation noise ≈ 1.4).
Validation rose (math500 40.6, ifeval 65.6, mmlu 49.1) and truncation fell further, but the private score did not follow: the
local proxy (62% GSM8K/MATH) over-predicted again. Across the 12 top leaderboard rows, private correlates negatively with
validation GSM8K (−0.81) and MATH-500 (−0.51) and positively with CommonsenseQA (+0.66), ARC-Easy (+0.69), IFEval (+0.65),
HumanEval (+0.57) and Winogrande (+0.53) (n = 12, noisy). A math-heavy proxy therefore mis-ranks edits on this stack; we
record this as the main negative finding of the experiment.

## Limitations
The proxy is math-heavy; self-distillation can shift answer style (checked with a stress set incl. IFEval/HumanEval/MBPP/MMLU).
The official private score decides.

---

## Second entry: v6 LoRA on cd0b95bb (2026-11-06, afternoon)

### Hypothesis
v3 raised strict locally but scored 32.4 private. Two suspected causes: (1) keeping the *shortest* correct trace taught the model to
shorten its reasoning, which lowered accuracy on math (proxy B: GSM8K −5, MATH-500 −3.8 for the shortest-trace v5); (2) the
training mix was 60% math while private looks broader. A median-length trace and a broader public train mix should keep reasoning
length and raise accuracy on the benches that track private (see `PROXYB-CALIBRATION.md`). Separately, poppingbean's
`cd0b95bb` (344c8376 + ×1.05 final-norm gain taken from our `27e49fd4`) scored 37.2 private; the norm gain and the LoRA touch
disjoint tensors, so the LoRA is applied on top of it.

### Method (differences from v3)
- Data (`gen_rft.py`, `PSET=v4`, ids from 200000): 8,074 public TRAIN prompts — CommonsenseQA train 1,400, ARC train 1,200,
  Winogrande train 1,200, IFEval-like (argilla/ifeval-like-data) 900, PIQA train 700, BoolQ train 600, OpenBookQA train 600,
  MMLU auxiliary_train 600, MBPP non-test 474, GSM8K train 400 — 2 samples each from 344c8376, 6,144-token cap; pooled with the
  v1/v3 generations → 35,644 generations.
- Selection (`select_rft.py --pick median --math_per 2`): per prompt, the median-length sample among correct, closed, ≥ 50
  thinking tokens, not truncated; two per GSM8K/MATH prompt → 11,763 traces (GSM8K 2,689, MATH 2,164, ARC 1,820, MMLU-aux 959,
  CSQA 946, Winogrande 892, IFEval-like 580, PIQA 538, BoolQ 509, OBQA 461, MBPP 205), mean 1,023 tokens.
- Training: same as v3 (r 16, alpha 32, lr 5e-5, 1 epoch, ~64k tokens/step, 203 steps, final loss ≈ 0.35, 54 min on one GPU),
  trained against 344c8376.
- Submission: the same LoRA tensors re-targeted at parent `cd0b95bb-c5c0-4656-b9be-ad6482f04075` (`rebase_lora.py`; only the
  manifest's parent fields change). The two edits touch disjoint tensors (final norm vs. layer projections).

### Evidence (local, advisory)
proxy400 strict, 2 seed sets × 3 samples, paired by prompt (A1 predictor, `agents/a1/predict.py` in our sprint handoff):

| model | vs pooled b8af1dbc ref | vs cd0b95bb | A1 decision |
|---|---|---|---|
| cd0b95bb (server 37.2) | +3.39 ± 1.21 | — | reject (screen 3.5) |
| v6 on 344c8376 | +5.68 ± 1.14 | +2.29 ± 1.10 | submit |
| **v6 on cd0b95bb (submitted)** | **+5.42 ± 1.17** | **+2.03 ± 1.03** | **submit** |
| v5 (shortest traces) on cd0b95bb | +4.01 ± 1.23 | +0.63 ± 1.19 | reject (acc among healthy −5.9) |

Stress set (265 prompts × 3 samples), accuracy, 344c8376 / v6 / v6-on-cd0b95bb: GSM8K 59/62/59, MATH-500 36/36/35, MMLU 43/49/48,
BBH 28/39/43, IFEval 58/62/62, HumanEval 32/31/39, MBPP 32/34/42; all 41/45/47. Truncation 6.8% / 5.7% / 6.5%. The 5 fixed
prompts close in 15/15 samples.

Proxy B (460 public test items weighted to private-correlated benches, 2 seed sets × 2 samples): v6 on 344c8376 57.7 vs parent
57.3 (no clear change); v5 58.4.

### Negative findings
- Shortest-trace selection on the broad mix (v5): BBH/MMLU/Winogrande up, GSM8K/MATH down, accuracy among finished answers
  −6; adds nothing on top of the ×1.05 norm gain (+0.6 vs cd0b95bb).
- Scaling the v5 LoRA ×1.5: worse than ×1.0 on proxy B (+0.6 vs +1.1).
- A1 is not reliable at the top: it passed v3 (server 32.4) and rejects cd0b95bb (server 37.2). Proxy B calibrated on 14 board
  checkpoints predicts private with LOO RMSE 2.2, the same size as private re-evaluation noise, so this entry is an option on a
  locally positive candidate, not a confirmed gain.

### Lineage and credit
`0ea62f40` → `a9340e14` → `b8af1dbc` → `344c8376` (SpacexCeo) → `cd0b95bb` (poppingbean: ×1.05 final-norm gain from SpacexCeo's
`27e49fd4`) → this entry (SpacexCeo: v6 LoRA).

### Verified result (v6 on cd0b95bb)
Submission `6f927dec-be15-4f7f-8fac-eb58c8c61751` → version `cfe73117-f16a-4858-a5a4-a7bdf39747b9`: private test **34.6**
(SE 2.13), validation 40.28, truncation 14.6%. Parent cd0b95bb scored 37.2 (SE 2.16); 344c8376 35.0. The LoRA did not add
on top of the norm gain on the server: the difference (−2.6) is about one standard error of the difference, so the honest reading
is "no measurable gain", consistent with v3 (32.4). Self-distillation on the model's own correct traces raised every local proxy
(proxy400 strict +5, stress +6) but did not move private in two entries; local proxies built from public test splits do not
capture what the private set rewards for this kind of edit.
