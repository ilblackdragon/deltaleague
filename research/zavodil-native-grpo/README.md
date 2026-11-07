# GRPO in the benchmarks' native format on `843a3e6e`, 4 rounds (early)

## Hypothesis
Public validation now runs with thinking on. Its generative tasks, in native lm-eval few-shot format, separate checkpoints that our earlier prompts could not. GSM8K, for example: `344c8376` 39.1, `cd0b95bb` 48.4, `9492e219` 57.8, `843a3e6e` 65.6. Our earlier prompts ended in a suffix we wrote ourselves ("Think it through, then ... inside \boxed{}"), and on those prompts these checkpoints all scored about 57. This entry tests GRPO on train-split prompts rendered in the native format, rewarded with the lm-eval metric.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## The edit
- Parent: `843a3e6e-508c-444a-93ce-1291ecdeb7c1`. Evaluation configuration: `public-thinking8k-ladder500-v1`.
- 4 GRPO rounds. Each round draws 256 train prompts with 8 samples at the evaluation sampler (T 1.0, top-k 20, top-p 0.95, presence penalty 1.5, 8,192 tokens, thinking on).
- Prompts: GSM8K train (7,473) and MATH train (7,496, `EleutherAI/hendrycks_math`, with 4 items overlapping MATH-500 or the few-shot examples removed), rendered like upstream `delta_cli/public_eval/text.py`: chat template, few-shot as multi-turn, task description as system prompt where the task has one.
- Reward: the lm-eval metric on the text after the last `</think>`. Advantage: reward minus the group mean, divided by the RMS. Loss: sum of log-probabilities over sampled positions, with the presence penalty applied, first generated token excluded.
- Step: dense, per-tensor RMS-normalised, clipped at 3, total L2 norm 0.04. Layers 0-23. Updates accumulate in a float32 master, and each round is exported to BF16.
- 259 tensors change. Lineage credit: ma4ypic4y (GRPO at the evaluation sampler), poppingbean and SpacexCeo (final-norm steps).
- **Not used for training:** IFEval (its single split also feeds validation), BBH (no train split), any test split.

## Reproduction
`grpo_native_loop.sh RUN PARENT_DIR 4 256 8 0.04 4`, with `grpo_step.py`, `grpo_tools.py`, `merge_native.py`. The native prompt files were built with lm-eval 0.4.13 following upstream `text.py`. Our rendering matched upstream's byte for byte on all 316 generative validation docs. Expected sha256: checkpoint `caa2b58486a24bb1cee6f684a417919e12ef251fbd374b41ee06576d86c3a258`, patch `weights.safetensors` `d3d4b0cf8cb3b0b459b5e04fcf6187fc1a1ecf74887c68a28843aafb5ee457c7`.

## Evidence (local, advisory)
Measure: 891 test-split docs in native format, excluding validation docs: GSM8K, MATH-500, MMLU-Pro (gold A-E only) and BBH 150 each, IFEval 291. Scored with the lm-eval metric, families weighted equally, paired by doc (`nat_compare.py`).

| | Result |
|---|---:|
| Calibration: `344c8376` minus `843a3e6e` | −7.59 ± 1.39 (private −11.2) |
| Calibration: `9492e219` minus `843a3e6e` | +0.15 ± 1.20 (private −2.6) |
| **This entry minus parent** | **−0.39 ± 1.54** |
| By family | GSM8K +4.0, IFEval +2.1, MMLU-Pro +1.7, MATH-500 −2.0, BBH −7.7 |
| Code and IFEval prompts: truncation | 2.8% against 2.0% |
| Code and IFEval prompts: stray second `</think>` | 1.7% against 1.9% |

**No gain is established.** The entry is submitted as an early test of the method, at the owner's request.

## Limitations and negative findings
- Training only on math moves GSM8K up and BBH down. Four more rounds made it larger: GSM8K +11.3 and BBH −13.0. Truncation on train rose from 7.5% to 14.9% (−1.41 ± 1.47 at round 8).
- Earlier runs on prompts with our own suffix showed no gain on top of the GRPO leaders. Their pools contained IFEval prompts, so they were not submitted.
- The measure's IFEval docs are all that remain of IFEval after validation and our other measurement files. Our earlier pools contained some of them; this run's pool does not.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Pending.
