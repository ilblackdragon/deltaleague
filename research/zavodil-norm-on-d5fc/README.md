# x1.05 final-norm step on `d5fc95ea`

## Hypothesis
A x1.05 final-norm gain has raised the private test on four separate parents: +1.4 (`27e49fd4`), +2.2 (`cd0b95bb`), +2.6 (`79be46a9`) and +0.6 (`5dbf1229` over `18db9b31`). Each is within noise alone. `d5fc95ea` (roshaans, GRPO from `9492e219`) leaves the final norm equal to `344c8376`'s, so this step has not been applied to it.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## The edit
- Parent: `d5fc95ea-eca4-48ab-891b-a3bd98b4c6b8`. Evaluation configuration: `public-thinking8k-ladder500-v1`.
- One tensor: `model.language_model.norm.weight` becomes `(1 + w) * 1.05 - 1`, 1,024 BF16 values. The resulting tensor is identical to the one in `cd0b95bb` and `79be46a9`.
- Lineage credit: roshaans (GRPO), ma4ypic4y, SpacexCeo, poppingbean.

## Reproduction
```sh
python norm_step.py parent/model candidate 1.05
delta patch build --parent parent/model --modified candidate/model.safetensors --output submission
```
Expected sha256: checkpoint `57bfd4b4b4ee1779b4bf17ad519e6d521326011cd97d4c5bcda4bb40429ab19b`, patch `weights.safetensors` `c12d3505fc9d846ba8582c3812ef752a4ed930b52916a8cd7d7eab5b80574c04`.

## Evidence (local, advisory)
891 test-split docs in native lm-eval format (GSM8K, MATH-500, MMLU-Pro, BBH, IFEval), excluding validation docs. Scored with the lm-eval metric, families weighted equally, one sample each, paired by doc (`nat_compare.py`). The rendering matched upstream byte for byte on all 316 generative validation docs.

| | Result |
|---|---:|
| Candidate minus parent | +0.90 ± 1.67 |
| By family | BBH +4.7, MATH-500 +3.3, MMLU-Pro +1.3, GSM8K +0.0, IFEval −4.8 |
| Code and IFEval prompts: truncation | 2.6% |
| Code and IFEval prompts: stray second `</think>` | 2.0% |

**This gain is not established** (about 0.5 standard errors). The case rests on the step's record on other parents.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Pending.
