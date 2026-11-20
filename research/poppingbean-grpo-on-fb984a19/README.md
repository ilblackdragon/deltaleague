# 8 GRPO steps on fb984a19 (roshaans' recipe, unchanged)

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, rank 1, private test 53.4).

## Hypothesis

roshaans' GRPO recipe (#58) raised `9492e219` to `d5fc95ea`. The recipe has three parts:

- an evaluation-shaped mix: 35% math, 30% multiple choice, 25% IFEval-like, 10% logic;
- a 6,144-token training cap;
- log-probs matched to the evaluation sampler.

Since then, every training-free edit on top of `fb984a19` has scored below it. Those edits include merges, SVD compression, norm and phase-gate scaling, and sign filtering. This entry tests whether more of the same on-policy GRPO still adds anything on the current leader. The truncation rate there is about 14%, and on MATH about a third of answers hit the cap.

## Method

- **Script:** `grpo.py` from #58, unchanged; credit roshaans. It is not copied here, so see that PR.
- **Run:** `run_grpo.sh` drives it with `--mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 --train-max-tokens 6144 --match-sampler --stop-on-decline 0.05`. Settings: 64 prompts × 8 samples per step, lr 1e-6, and a strict reward (correct, `</think>` closed, not truncated).
- **Training data:** public train splits only. The math items used by the local probe are excluded.
- **Stop:** planned for 24 steps. The stop rule (mean reward of the last 4 steps more than 0.05 below the first 4) fired at **step 8**: 0.618 → 0.522. Per-family rewards swing widely between steps, because each step sees only a few prompts per family. So the decline is probably mostly batch noise.
- **Compute:** one rented RTX 6000 Ada (48 GB) with vLLM 0.30, about 4.3 min per step. The pod cost was about $6–8 in total, covering setup, training and both evaluations.
- **Patch:** `delta patch build` against `fb984a19` changes 263 tensors, all under `model.language_model`. `delta validate` passes.

| Artifact | sha256 |
|---|---|
| step-8 weights | `5e5cf6fa0db852b3a811083444862e2fdfcd561a1f286cedf103d22e09c61099` |
| patch weights | `2a4992639f4a1188411ca2fee4a31751442fb2b78a9ea51d6042802601abac3e` |
| patch manifest | `c097df6f1a575bd7d000379b4adc1cb891a1263ccd6d9ba834e69806c372931b` |

Training log: `results/train_log.jsonl`.

## Local evidence (advisory)

**Math probe** (`probe.py` from #58): 256 gsm8k and 256 MATH train items held out from training, run at the evaluation sampler with an 8k cap. The same seeds were used for both models.

| | `fb984a19` | step 8 |
|---|---|---|
| gsm8k | 75.4 | 73.0 |
| MATH | 40.2 | 40.6 |
| all | 57.8 | 56.8 |
| truncated | 17.0% | 15.4% |
| median tokens | 1,892 | 1,853 |

The paired difference is **−0.98 ± 1.78** (39 wins, 44 losses), a statistical tie.

**Public groups:** the 14 text groups and 2 coding groups of the public suite (BFCL excluded), run at the evaluation sampler (thinking on, 8k cap, presence 1.5) and paired on identical documents.

| Group | `fb984a19` | step 8 | diff |
|---|---:|---:|---:|
| arc_challenge | 26.6 | 26.6 | +0.0 |
| arc_easy | 68.8 | 68.8 | +0.0 |
| bbh | 33.3 | 27.8 | -5.6 |
| boolq | 64.1 | 64.1 | +0.0 |
| commonsense_qa | 57.8 | 57.8 | +0.0 |
| gsm8k | 62.5 | 59.4 | -3.1 |
| hellaswag | 43.8 | 43.8 | +0.0 |
| ifeval | 71.9 | 76.6 | +4.7 |
| math500 | 59.4 | 57.8 | -1.6 |
| mmlu | 48.2 | 48.2 | +0.0 |
| mmlu_pro | 32.9 | 28.6 | -4.3 |
| openbookqa | 29.7 | 29.7 | +0.0 |
| piqa | 71.9 | 71.9 | +0.0 |
| winogrande | 60.9 | 59.4 | -1.6 |
| humaneval | 28.1 | 28.1 | +0.0 |
| mbpp | 25.0 | 21.9 | -3.1 |
| **mean (16 groups)** | **49.05** | **48.14** | **-0.91** |

Reading:

- The likelihood-scored groups (ARC, BoolQ, CSQA, HellaSwag, MMLU, OBQA, PIQA) are identical.
- Across the 7 generative groups, step 8 averages **-1.9**. IFEval is the only gain (+4.7, consistent with 25% IFEval-like training data). BBH, MMLU-Pro, gsm8k and MBPP move down.
- No group collapses: the largest drop is −5.6 on 54 BBH items, which is within one standard error of a 54-item group.
- Overall, the local evidence leans slightly **negative**: −0.9 on the public groups and −1.0 ± 1.8 on the math probe. This entry is submitted as a measured test of further GRPO on the leader, not as an expected improvement.

## Evaluation notes

- **IFEval fingerprints:** the official `delta public-eval run` aborted on the parent at IFEval. In this environment, the IFEval documents no longer match the suite's pinned document fingerprints ("selected documents changed").
- **Workaround:** `pe_groups.py` runs the same text and code groups through the evaluator's own `text.run` and `coding.run`, at the evaluation config's sampler. It skips only the IFEval fingerprint check and BFCL. Both models see identical documents, so the comparison stays paired. It is not an official public-validation score.

## Official result

Submission `079a9847-702b-4eb3-8bec-058ec7039036`, version `1b23b576-64fb-4990-b4f5-5a3c3ce71e30`, verified.

| | `fb984a19` (parent) | **This entry (step 8)** |
|---|---|---|
| Private test (500 tasks) | 53.4 | **51.5** (rank 5, tied) |
| Public validation | 47.40 | 47.01 |
| Test truncation rate | 14.2% | 14.4% |

The private test fell by 1.9 points, matching the direction of the local evidence: −0.9 on the public groups and −1.0 on the math probe. With a test SE of about 2.2 per entry, the drop is not significant on its own. Still, all three signals agree that 8 more GRPO steps with this recipe do not improve `fb984a19`. The IFEval gain seen locally did not lift the overall private score, and truncation did not fall.
