# A second x1.05 final-norm step on `79be46a9`

## Hypothesis
A x1.05 final-norm gain has raised the private test in three separate entries: +1.4 (`27e49fd4`), +2.2 (`cd0b95bb`) and +2.6 (`79be46a9`). Each is within noise alone. This entry tests whether one more step, x1.1025 in total over `9492e219`, keeps helping.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## The edit
- Parent: `79be46a9-cba5-4535-9076-1d97ba749b29`. Evaluation configuration: `public-ladder500-8k-128-v1`.
- One tensor: `model.language_model.norm.weight` becomes `(1 + w) * 1.05 - 1`, 1,024 BF16 values.
- Lineage credit: ma4ypic4y (GRPO), poppingbean, SpacexCeo.

## Reproduction
```sh
delta competition init qwen35-08b-v1 --parent 79be46a9-cba5-4535-9076-1d97ba749b29 --output leader
python norm_step.py leader/model candidate 1.05
delta patch build --parent leader/model --modified candidate/model.safetensors --output submission
delta validate submission
```
Expected sha256: checkpoint `1756ac04b27de7312f931a6380af3d0f00fcae918282c33b78cdf88e9b494ace`, patch `weights.safetensors` `b50d2aa7a310dc3d530ac74d0a39dfb7240b64fc4270dc9cddac962ad92165d2`. Format sparse-diff, one tensor.

## Evidence (local, advisory)
Generations with the competition's thinking-mode settings on an RTX 4080 with vLLM 0.30.0, scored with `delta_cli.scoring.score_answer`. Public test or validation splits, measurement only. 14 benchmark families with equal weight: MATH-500, GSM8K, MMLU-Pro and ARC-Challenge (320 prompts, 3 samples), and 10 more families (`build_wide.py`, 30 prompts x 2 samples and 150 prompts x 4 samples). Paired by prompt with `compare_official.py`. The parent's weights match a local rebuild except for 4 entries, each off by less than 1e-12, and that rebuild stands in for the parent on 3 seeds. The candidate ran one seed.

| Candidate minus parent | Result |
|---|---:|
| 14 families | **+0.28 ± 0.52** |
| x1.025 instead, same measure | −0.28 ± 0.48 |
| Truncation on the 320 prompts | 11.8% against 11.4% to 12.2% |
| Code and IFEval prompts: truncation | 2.0% against 1.3% |
| Code and IFEval prompts: stray second `</think>` | 1.9% against 3.0% |

**This gain is not established.** It is about 0.5 standard errors. The entry is a test of the private response to the step, not a demonstrated improvement.

## Limitations and negative findings
- The same measure gives the x1.05 step from `9492e219` to `79be46a9` only +0.41 ± 0.46, against +2.6 on the private test. Repeated samples suggest a sampling error near 1.3 for one 500-task run and 1.8 for a difference, so the two agree within noise.
- The measure's prompts are a guess at the private format, and its benchmark mix is unknown relative to the private test. It reproduces platform truncation closely (r 0.995 over 10 checkpoints, Original and `632210ae` the largest misses) but not the order of private scores among the top entries.
- Continuing the leader's GRPO delta does not help: x1.5 gives −1.44 ± 1.34 and x2 gives −3.84 ± 1.63, and first-token `</think>` rises from 19% to 38% and 72% on the 320 prompts.
- Blocking `</think>` as the first generated token with a rank-1 layer 23 edit gives −0.16 ± 1.59 and breaks code prompts: stray `</think>` 1.9% to 22.4%, truncation 1.3% to 7.4%.
- Further GRPO on this parent, with the official score as reward over all families: −0.31 ± 0.89 after 5 rounds on 3 seeds, compared with `9492e219` on the same seeds. It shortens traces (truncation 6.9% to 5.8%) without raising accuracy.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Verified (official): private test 46.2, tied at official rank 1 with the parent, validation 44.34, truncation 10.8%. Parent `79be46a9`: 46.2, 44.94, 12.4%. Version `843a3e6e-508c-444a-93ce-1291ecdeb7c1`.

The private score did not change, so this entry shows no gain over its parent. Truncation fell 1.6 points and validation fell 0.60. With a sampling error near 1.8 on a difference, none of these is established. Validation moved most on MBPP (−9.4), IFEval (−6.2), MATH-500 (−6.2) and MMLU-Pro (+4.3).
