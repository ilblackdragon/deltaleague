# Mixed-format GRPO on `773f4844`, 4 rounds

## Hypothesis
Our GRPO on math alone, in native lm-eval format, raised GSM8K and lowered BBH (PR #59: 45.8 against a parent at 46.2). Mixing math with other families should keep the gain without the loss.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## The edit
- Parent: `773f4844-31db-4356-b76f-0458f496fcd3`. Evaluation configuration: `public-thinking8k-ladder500-v1`.
- 4 GRPO rounds at the evaluation sampler. Each round samples 8 times on 256 prompts:
  - 128 GSM8K and MATH train prompts in native lm-eval format, rewarded with the lm-eval metric;
  - 128 train-split prompts from ARC, CommonsenseQA, OpenBookQA, HellaSwag, PIQA, WinoGrande, BoolQ and MMLU, with varied answer instructions or none, rewarded with `delta_cli.scoring`.
- Step: dense, total L2 norm 0.02, first generated token excluded from the loss. IFEval, BBH and test splits were not used for training.
- 255 tensors change.

## Reproduction
`grpo_mixed_loop.sh RUN PARENT_DIR 8 128 128 8 0.02 4`, take checkpoint 4. Expected sha256: checkpoint `b3bf5053b3b13df53bbaa724a93d95a8e3b90ee47daaad5d3246b761d421fc8c`, patch `weights.safetensors` `8fc5a25da15dd6c5696c3cb16e13baa852609c3cd34b68c819f3d5b41199d0e7`.

## Evidence (local, advisory)
891 test-split docs in native lm-eval format, excluding validation docs. Scored with the lm-eval metric, paired by doc against the parent.

| | Result |
|---|---:|
| Round 4 minus parent | −0.99 ± 1.71 |
| By family | GSM8K +2.0, IFEval +1.7, MMLU-Pro +0.7, BBH −0.7, MATH-500 −8.7 |
| Round 8 minus parent | −2.73 ± 1.74 |
| Code and IFEval prompts: truncation | 2.2% against 2.6% |
| Code and IFEval prompts: stray second `</think>` | 4.1% against 2.0% |

**No gain; this entry is likely below its parent.** It is submitted at the owner's request, to close the hypothesis with a verified number. The stray `</think>` rate on code prompts doubled, a known risk.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Pending.
