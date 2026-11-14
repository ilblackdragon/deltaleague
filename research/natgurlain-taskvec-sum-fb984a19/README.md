# Task-vector sum of two truncation-reducing descendants of fb984a19

Research by natgurlain with an AI agent (Claude Code). Available during research: public leaderboard rows, public checkpoints, public research PRs and the `delta` CLI. No private test material.

- **Parent:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, rank 1, private 53.4, truncation 14.2%), content hash `bd3c11a04a4d1691c1de6b7afadd18d8a56c232d29aa8959fec41819ff43ddfd`
- **Evaluation configuration:** `private1000-v6-thinking8k-20261111` (`delta-public-v1`)
- **Submission:** `5b41e3ce-e5a7-4801-9eeb-0c01284ad232` (version `8ae89537-b257-4180-89f3-0fe6701994ae`), queued; no verified scores yet

All statements about quality below are local and advisory unless marked verified.

## Hypothesis

Two published descendants of `fb984a19` each lowered the truncation rate on the private test without a resolvable change in score:

| Version | Author | Private | Truncation | What it changes against `fb984a19` |
|---|---|---:|---:|---|
| A `5d4b0b84-279e-4451-a002-83ae770a143e` | poppingbean (#77) | 53.3 | 12.6% | two-seed soup of length-shaped GRPO; 255 tensors, 4.5% of their elements |
| B `a075787e-9d42-4470-af1e-6674cdd3bf55` | iJ03l (#68) | 52.9 | 12.4% | final RMSNorm gain of about x1.0115, plus a small MLP change; 61 tensors |

The two changes are nearly independent (see below). If their effects on truncation add, the sum
`theta = P + (A - P) + (B - P)` should truncate less than either while keeping accuracy. Unfinished answers score zero, so truncation is the largest visible lever left on this parent.

A sum is used instead of an average because both deltas are mostly single bf16 steps. Half a step is not representable, so an average would round most of each delta away.

## Method

`merge.py` computes `P + wa*(A - P) + wb*(B - P)` per tensor in float64 with `wa = wb = 1` and stores the parent's dtypes. There is no training, no GPU use and no randomness, so there are no seeds. Non-float tensors are required to be identical in all three checkpoints.

## Measured properties of the two deltas

From `merge_report.json` and a separate overlap check:

- A changes 255 tensors, B changes 61; 60 are shared, and the final norm is changed only by B.
- B's final-norm ratio to the parent: mean 1.01148 (min 1.00966, max 1.01575).
- Excluding the final norm: `|A - P|` = 0.0206, `|B - P|` = 0.0014, cosine −0.005. On the 2,031,250 elements both change, signs agree 48.7% of the time.
- The result changes 256 tensors and 33,808,198 elements against the parent.

So the candidate is, to a close approximation, A plus B's final-norm step. B's MLP change is about 15 times smaller than A's and uncorrelated with it.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output exp-leader
delta model pull 5d4b0b84-279e-4451-a002-83ae770a143e --output pulls/5d4b0b84
delta model pull a075787e-9d42-4470-af1e-6674cdd3bf55 --output pulls/a075787e
python merge.py exp-leader/model/model.safetensors pulls/5d4b0b84/model.safetensors \
  pulls/a075787e/model.safetensors exp-leader/edited.safetensors --report merge_report.json
cd exp-leader
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
```

Environment: `deltaleague` 0.3.0, Python 3.12.8, torch 2.15.0, CPU only for the merge.

**Expected sha256:**

- parent `model.safetensors`: `fc2a115f0507251baebef840d727b3bbf626ee861e8ed4d6bd33c0e7f81441e6`
- A `model.safetensors`: `873170be711182d91c9403a68c153e5455e80a863e29da5332f7604698da25bd`
- B `model.safetensors`: `1e3a0f5678064e7cbb940f5eb0c69d3b4277dbc89b6613e53ba312cf85ff7efe`
- candidate `edited.safetensors`: `d9cc5dd3d36a4c2ad1328a7ad9d15415cf8a6c6355fb3d1364357ada0888491b`
- patch `weights.safetensors` (sparse, 256 tensors): `13cf2e8926418e857c3cf1ee358ef26a561dc6f63f27799019c7e4143cbad3fc`
- `manifest.json`: `4f25157c465e563efe08efd53f12190191723801ddc10e5f4c1ff5ab20a6af47`

`delta validate` passes.

## Evidence

**There is no local accuracy evidence for this candidate.** The full pinned public suite was not run for the parent or the candidate. Earlier PRs on this parent (#69, #74, #77) report that local probes did not predict the private ordering at this noise level.

The only local check is for breakage, using `delta run` on an RTX 3090 with the candidate reconstructed from the patch (`--patch submission`):

- Three short prompts without thinking: the candidate's output is identical to the parent's on all three.
- One arithmetic word problem with thinking on (3,000-token cap): both close `</think>` and answer 160, which is correct; neither is truncated.

Four prompts say nothing about accuracy or truncation rates.

## Limitations

- The inputs were chosen by their private-test truncation rate and score, both of which are noisy; differences among the top rows are within the leaderboard's standard error of about 1.6.
- The additivity of the two effects is assumed, not measured.
- The final-norm step has a mixed record. On the current leaderboard, pure x1.05 steps on five earlier parents sit 2.0 to 3.7 points above their parents (`773f4844`, `843a3e6e`, `79be46a9`, `cd0b95bb`, `27e49fd4`) and one sits 0.7 below (`630aa8d6`, on the original model). On this parent, whose lineage already contains several such steps, both entries that raise the final norm score below it: B itself (−0.5) and `bb58440c` (x1.042 combined with a gate boost, −2.6). A further x1.0115 step on top of A may therefore not help.
- A single private-test result cannot establish a gain or a loss of less than about 2 points.

## Credit

SpacexCeo (`fb984a19`), poppingbean (#74, #77, on roshaans' GRPO recipe from #58) and iJ03l (#68) produced every weight change used here. This experiment only combines them.

## Boundaries

- Only this experiment directory changes.
- No weights, credentials, private test material, signed URLs or raw generation logs are included.
