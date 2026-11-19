# GRPO at the evaluation sampler, continued from 9492e219 (work in progress)

Parent: `9492e219-6f07-4bdc-b863-e86274ee62c2` (ma4ypic4y, "GRPO at the evaluation sampler, 20 steps", itself on our
`344c8376`). Credit: the idea of reward-optimising at the server sampler is theirs; no recipe was published, this is an
independent implementation.

## Why
- Across the 34 scored versions, private test correlates −0.89 with the server truncation rate and only 0.15 with the
  validation score. 9492e219 has the lowest truncation (0.116) and keeps accuracy on finished answers.
- Weight diff 9492e219 vs 344c8376: 271/488 tensors (every linear weight of all 24 layers + embeddings), relative change
  0.056%, high-rank updates — full-parameter training, not LoRA or a direction edit. `</think>` / EOS rows barely move.
- Our surgical edits on cd0b95bb (doubt / re-check / norm remixes, anti-enumeration and anti-waffle direction ablations)
  all failed to beat their parent locally: they move failures between truncation modes without adding reasoning.

## Method (`grpo.py`, `make_pool.py`)
- Rollouts: vLLM with the server sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 8192 tokens),
  8 samples × 32 prompts per step from public TRAIN splits only (GSM8K, MATH L1–4, ARC, CSQA, Winogrande, BoolQ, PIQA,
  OBQA, MMLU auxiliary_train), weighted toward intermediate difficulty.
- Reward: correct AND `</think>` closed AND not truncated.
- Update: on-policy, group-normalised advantages, one AdamW step per GRPO step, full parameters except the tied
  embedding, fp32 master weights with bf16 autocast; checkpoint written in bf16 with the parent's tensor keys.
- `collect_p.py`, `dirs_p.py`, `tf_p.py`, `build_p.py`: the direction-ablation study on cd0b95bb that was killed
  (negative result, kept for reproducibility).

Results, evaluation IDs and seeds will be added before any submission.

## Night results (2026-11-07, proxy400 = math500/gsm8k/mmlu_pro/arc_c, 3 samples/prompt, server sampler)
Reference 9492e219: seed set 1+2 acc 45.4, truncation 11.9%, accuracy on finished answers 53.0.

| candidate | result | verdict |
|---|---|---|
| GRPO run1: lr 5e-6, group-std advantage | multi-`</think>` hack from step 5 (39% → 100% of rollouts), thinking median 490 → 36 tokens by step 30 | collapsed |
| GRPO lr 1e-5 | 95% truncated rollouts at step 3 | unstable |
| GRPO variant B: lr 3e-6, mean advantage, DAPO overlong shaping | step 15: acc 43.1, **truncation 33.0%**, acc on finished 63.9 — learns to think longer | worse |
| GRPO + Kimi in-group length reward (w 0.3 / 0.15), hard pool | skipped thinking in 50–70% of rollouts, multi-close returns | collapsed |
| ExPO 9492e219 + a(9492e219 − 344c8376), a 0.5 / 1.0 | acc −5.6 ± 1.7 / −13.3 ± 1.9, truncation −2 | worse |
| **9492e219 + final norm ×1.05** | s1+s2: acc **46.8** (paired **+1.41 ± 1.11**), truncation 12.0%, acc on finished 55.4 | submitted (probe) |
| 9492e219 + final norm ×1.025 | s1: acc −1.56 ± 1.47 | not submitted |

The ×1.05 final-norm gain is poppingbean's edit (ilblackdragon/deltaleague#43, which put it on our 344c8376); on the server
it gave +1.4 on b8af1dbc (27e49fd4) and +2.2 on 344c8376 (cd0b95bb). Here it is applied to ma4ypic4y's GRPO model.
Patch: one tensor (`model.language_model.norm.weight`, (1+w) × 1.05), parent 9492e219-6f07-4bdc-b863-e86274ee62c2.
Seed set 2 of the candidate ran on an RTX 3090 while the reference ran on an RTX 5070 Ti (vLLM numerics can differ).
Lesson: continuing GRPO from 9492e219 with our prompt pools is unstable within 5–15 steps; format guards (single
`</think>`) are needed from step 1 and length shaping in either direction is harmful.

## Best GRPO checkpoint: run5 step 10 (second candidate)
Config: parent 9492e219, pool = public TRAIN math + MC (6300 prompts, `make_pool.py` ∪ `make_pool_hard.py`), 32 prompts × 8
samples per step at the server sampler, reward = correct ∧ single `</think>` ∧ closed ∧ not truncated (no length shaping),
mean-centred advantage (no std), one AdamW step per batch, lr 2e-6 with 2-step warmup, embeddings frozen, fp32 masters.
Checkpoints every 5 steps; step 10 selected on proxy400 (step 20 drifts to longer thinking: truncation 24%).

| proxy400 (3 samples/prompt) | acc | truncation | acc on finished |
|---|---|---|---|
| 9492e219, seed sets 1+2 | 45.4 | 11.9% | 53.0 |
| run5 step 5, seed set 1 | 47.7 (paired +1.04 ± 1.57) | 11.8% | 53.6 |
| **run5 step 10, seed sets 1+2** | **50.2 (paired +4.79 ± 1.16)** | 14.4% (+2.5 ± 0.8) | 58.4 |
| run5 step 20, seed set 1 | 46.0 (−0.62) | 24.0% | 59.7 |
| run6 (hard pool only) step 10, seed set 1 | 45.8 (−0.83) | 20.6% | 57.0 |

Per benchmark (step 10 vs 9492e219, pooled): math500 36.4 vs 37.5, gsm8k 70.2 vs 67.1, mmlu_pro 35.6 vs 26.2, arc_c 80.8 vs 64.2.
Risk: private correlates with truncation across the board, and this checkpoint truncates 2.5 points more on proxy400.
Patch: 264 modified tensors vs 9492e219 (`delta patch build`), checkpoint sha256 b86ac75f…0e03.
Stress set (265 public prompts + 5 fixed, 3 samples, server sampler): 9492e219 45.4 → run5 step 10 **50.4**
(bbh 40.0 → 52.5, mmlu 36.7 → 47.5, gsm8k 64.2 → 68.3, math500 30.0 → 34.2, ifeval 70.0 → 73.3, humaneval 38.9 → 37.8,
mbpp 41.1 → 40.0; truncation 5.4% → 6.4%, closed 94.7% → 97.0%). Adding the ×1.05 norm on top of step 10 does not help
(proxy400 s1+s2: −0.99 ± 1.05 vs step 10).

Robustness checks (proxy400 s1+s2 vs 9492e219): the same recipe with another seed (run7, step 10) gives acc +2.86 ± 1.14
but truncation +8.75 — the length drift depends on the seed; a uniform soup of run5/run7 step 10 gives acc +3.59 ± 1.08,
truncation +5.62. run5 step 10 has the best accuracy/truncation trade-off and is the submitted candidate.

## Verified result
Submission: GRPO run5 step 10 on 9492e219 (version `18db9b31-ffe9-427e-9e82-cfb66a0bfc80`, config public-thinking8k-ladder500-v1).
**Private test 47.4 (SE 2.24), truncation 0.132, validation 42.82 — rank #1 at scoring time** (vs 9492e219 43.6;
79be46a9 = 9492e219 + norm ×1.05: 46.2; d5fc95ea = 9492e219 + 36 GRPO steps: 47.2).

## Second submission: run5 step 10 + final norm ×1.05
Patch vs 9492e219: the 264 tensors of the run5 step-10 patch plus `model.language_model.norm.weight` with (1+w) × 1.05
(poppingbean's edit, ilblackdragon/deltaleague#43). Server history of a single ×1.05 step: +1.4 (b8af1dbc), +2.2 (344c8376),
+2.6 (9492e219 → 79be46a9). Local proxy400 s1+s2 on our checkpoint: −0.99 ± 1.05 vs step 10 (inside noise). Submitted as a
test of whether the norm gain still adds after further GRPO.
**Verified: private 48.0 (version `5dbf1229-af50-443f-8cd0-117f4688f4ed`), truncation 0.134 — rank #1**, +0.6 over the
step-10 checkpoint alone (47.4). The single ×1.05 final-norm step has now added on the server on four parents.

## Combining two independent GRPO continuations of 9492e219 (prepared, CPU-only analysis)
Two GRPO runs from the same parent with different recipes: ours (run5 step 10, verified 47.4) and roshaans' d5fc95ea
(ilblackdragon/deltaleague#58, verified 47.2). CPU weight analysis vs 9492e219: both change 262 shared tensors, delta
norms 0.103 (ours) and 0.091 (roshaans), **cosine 0.024** overall (0.01–0.03 in every tensor family) — nearly orthogonal,
largely because both deltas are dominated by bf16 single-ulp rounding moves.
Candidates (patches vs 9492e219, final norm ×1.05 applied on top, as on four previous parents):
- **soup**: 9492e219 + (Δ_ours + Δ_roshaans)/2 + norm ×1.05 (265 tensors)
- **sum**: 9492e219 + Δ_ours + Δ_roshaans + norm ×1.05 (268 tensors)
No GPU evaluation was run for these two (no compute available); prior evidence: a soup of two of our own seeds landed
between them on the proxy, and scaling a single GRPO delta by 1.5 (ExPO) cost −5.6 proxy accuracy. Lineage credit:
roshaans (d5fc95ea), ma4ypic4y (9492e219), poppingbean (×1.05 norm step), zavodil (norm on d5fc95ea, ilblackdragon/deltaleague#60).

## Third-day entry: three-source TIES merge (CPU)
`ties.py` (TIES without trimming: per element, elect sign = sign(sum of deltas), merged delta = mean of the agreeing,
non-zero deltas; float64 math). Run on two sources it reproduces poppingbean's ac147fa6 merge bit-exactly
(sha256 5990f40263fc…, ilblackdragon/deltaleague#62). Here a third source is added: roshaans' 0c834b41
(d5fc95ea + 9B-teacher distillation + 30 GRPO steps, verified 48.8). Base 9492e219; sources d5fc95ea, 18db9b31 (ours),
0c834b41; final norm = the ×1.05 tensor (from ac147fa6 / 79be46a9). Merged checkpoint sha256 72e4e412692838de…;
layer delta |Δ|/|θ| 3.9e-4 vs 3.4e-4 for the two-source merge, cosine 0.82 with it. No GPU evaluation (no compute);
a lottery-level bet that the third run adds signal under sign election. Lineage credit: roshaans, ma4ypic4y, poppingbean, zavodil.
**Verified: private 46.0 (version 0d7af43b-7c06-495c-8cc9-79217b81622c) — worse than the two-source merge (50.4).**
Likely cause: 0c834b41 already contains d5fc95ea's delta, so the third "vote" double-counts roshaans' lineage and its extra
distillation moves dilute the sign election; unevaluated CPU merges are a lottery at this noise level (SE ≈ 2.2).

## GRPO on top of the leader ac147fa6 (run9, aborted)
Same recipe as run5 (lr 2e-6, mean advantage, single-`</think>` guard, 32×8, union pool), parent ac147fa6. Steps 1–4 were
healthy (rollout reward 0.54 → 0.80). After step 4 one element of the frozen tied embedding (row 185382) became non-finite
in the saved checkpoint, so every later rollout ran to 8192 tokens (reward 0). The embedding is frozen in the trainer and
the same code ran 12–30 steps cleanly on four other GPUs, so this looks like a memory fault on that rented card.
Repaired step-4 checkpoint (parent embedding restored) vs ac147fa6 on proxy400 seed set 1: acc +0.73 ± 1.28, truncation
+1.35 ± 1.00 — no clear gain after 4 steps; not submitted. Practical note: check `torch.isfinite` on every saved checkpoint.
