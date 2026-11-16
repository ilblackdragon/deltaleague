# Fair Cast GRPO on fb984a19: stop losing the updates to bf16 rounding

Research by natgurlain with an AI agent (Claude Code). Available during research: public leaderboard rows, public checkpoints, public research PRs, public training splits of open datasets, arXiv papers and the `delta` CLI. No private test material.

- **Parent:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, private 53.4), content hash `bd3c11a04a4d1691c1de6b7afadd18d8a56c232d29aa8959fec41819ff43ddfd`
- **Evaluation configuration:** `private1000-v6-thinking8k-20261111` (`delta-public-v1`)
- **Submission:** not submitted yet

All statements about quality below are local and advisory unless marked verified.

## Hypothesis

The shared GRPO recipe on this lineage (#58, #74, #77, and my #81) keeps an fp32 master, takes AdamW steps at learning rate 1e-6 and exports to bf16 by round-to-nearest for the vLLM sampler and for the patch. Near a weight of magnitude w the bf16 grid spacing is about w/128, so an update smaller than w/256 is discarded by the export. At 1e-6 almost every update is that small.

Measured on my #81 run (same recipe): only 3.1%, 4.6%, 5.7% and 6.5% of the bf16 weights differed from the parent after 4, 8, 12 and 16 optimizer steps. The sampler, and the submitted model, saw a policy that had barely moved, while the optimizer believed it had.

Two papers describe this failure and a fix: "Fair Cast" (arXiv 2610.41073) and arXiv 2611.13545. The fix has two halves, and both are needed (the first paper reports collapse with the first half alone):

1. export with **stochastic rounding** (round up with probability equal to the fractional position between the two neighbouring bf16 values), once per optimizer step, so the served weights are an unbiased image of the master;
2. compute the **loss on that served copy**, then apply the gradient to the fp32 master.

If the recipe was limited by rounding rather than by the learning signal, the same budget should now produce a measurable gain.

## Method

`grpo.py --fair-cast` (functions `sr_cast` and `export_state`, and the block guarded by `a.fair_cast` in the training loop):

- `sr_cast` adds 16 uniform random bits to the fp32 bit pattern and truncates the low 16 bits. Values already on the bf16 grid are unchanged and the expectation equals the fp32 value. Checked on random tensors before training.
- Before each backward pass the master parameters are moved to CPU and the served (rounded) file that the sampler used is loaded into the model; after the backward pass the master is restored and the optimizer steps on it. A new stochastic export is written once per step.

Everything else is the #58 recipe with the #74 length shaping, as in #81: AdamW 1e-6, bf16 autocast, `--match-sampler`, `--len-shape 3000`, 6,144-token training cap, the evaluation sampler for rollouts (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5).

Run: 60 steps of 32 prompts x 8 samples, seed 20261116, from the parent. Training pool `mined/mix.jsonl` (`make_mix.py`): the 1,330 "tail" prompts of #81 (prompts where the parent's 4 samples disagree or run long) plus 443 other mined prompts, 1,773 in total (math 687, multiple choice 507, instruction following 415, logic 164).

One run, one seed. Checkpoints were saved every 10 steps; only step 60 was probed, so there was no selection among checkpoints.

## Evidence

**Served weights now move.** Fraction of bf16 weights that differ from the parent (`results/train_log.jsonl`, `changed_vs_init`): 4.2% after step 1, 9.7% after 10, 12.0% after 20, 13.5% after 30, 14.6% after 40, 15.6% after 50, 16.4% after 60. Round-to-nearest in #81 reached 6.5% after 16 steps of four times the batch.

**Held-out probe.** The parent and the step-60 checkpoint are sampled 4 times on training-split prompts with the evaluation sampler and the same request stream, and compared per prompt (`cmp.py`, paired bootstrap). No probe prompt is in the training pool (checked by key and by prompt text).

| | Parent | Step 60 | Difference | 95% interval | Truncated at 8,192 | Mean tokens |
|---|---:|---:|---:|---|---:|---:|
| set 1 (1,000 prompts) | 65.33 | 67.97 | +2.65 | [+1.15, +4.25] | 7.15% → 4.97% | 2,028 → 1,969 |
| set 2 (1,000 fresh prompts) | 65.58 | 67.38 | +1.80 | [+0.25, +3.33] | 6.38% → 4.33% | 1,928 → 1,885 |
| pooled (2,000) | 65.45 | 67.67 | +2.23 | [+1.19, +3.29] | 6.76% → 4.65% | 1,978 → 1,927 |

By pool, both sets together: multiple choice +1.17 ± 1.08, instruction following +2.35 ± 1.18, logic +6.12 ± 1.92, math +1.93 ± 0.76 (n = 600, 500, 200, 700).

Unlike #81, the gain is not a length effect: answers are only about 3% shorter (#81: 20% shorter and no verified gain).

**Pinned public suite** (1,110 items, one sample each, run locally; `results/public_suite.json`): parent 47.43, step 60 47.90.

| Benchmark | Items | Parent | Step 60 | Difference |
|---|---:|---:|---:|---:|
| arc_challenge | 64 | 26.6 | 26.6 | +0.0 |
| arc_easy | 64 | 67.2 | 68.8 | +1.6 |
| bbh | 54 | 31.5 | 29.6 | −1.9 |
| boolq | 64 | 64.1 | 64.1 | +0.0 |
| commonsense_qa | 64 | 59.4 | 56.2 | −3.1 |
| gsm8k | 64 | 60.9 | 67.2 | +6.2 |
| hellaswag | 64 | 43.8 | 45.3 | +1.6 |
| ifeval | 64 | 76.6 | 78.1 | +1.6 |
| math500 | 64 | 54.7 | 62.5 | +7.8 |
| mmlu | 114 | 47.4 | 48.2 | +0.9 |
| mmlu_pro | 70 | 41.4 | 34.3 | −7.1 |
| openbookqa | 64 | 28.1 | 28.1 | +0.0 |
| piqa | 64 | 71.9 | 71.9 | +0.0 |
| winogrande | 64 | 60.9 | 62.5 | +1.6 |
| humaneval | 32 | 34.4 | 31.2 | −3.1 |
| mbpp | 32 | 25.0 | 18.8 | −6.2 |
| bfcl_single_turn | 72 | 56.9 | 62.5 | +5.6 |
| bfcl_multi_turn | 32 | 3.1 | 6.2 | +3.1 |

Each row is 32 to 114 single samples, so one row moves by 6 to 9 points on noise alone and the overall difference of +0.47 is well inside the noise of this suite. The suite does not contradict the probe and does not confirm it. The two code rows are down by one and two items; code is not in the training pool.

**Training reward** is flat within noise (means over ten steps: 0.549, 0.523, 0.542, 0.594, 0.607, 0.557); each step draws different prompts.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output exp-leader
ln -s ../exp-leader/model parent                     # inside the directory holding these scripts
. ./gpu_lib.sh; mkdir -p mined probe
probe leader parent                                  # probe set 1: 1,000 prompts x 4, seed 1
$PY mine.py parent mined/leader.jsonl --prompts 3000 --seed 2 --k 4 --exclude probe/leader.jsonl
$PY select_tail.py mined/leader.jsonl mined/tail.jsonl 4096
$PY make_mix.py                                      # mined/mix.jsonl, 1,773 prompts
./night_e.sh                                         # the Fair Cast run, runs/fc, checkpoints every 10 steps
probe fc_s60 runs/fc/step60 && $PY cmp.py probe/leader.jsonl probe/fc_s60.jsonl
$PY mine.py parent probe/leader_p2.jsonl --prompts 1000 --seed 3 --k 4 --exclude probe/leader.jsonl --exclude mined/leader.jsonl
$PY mine.py runs/fc/step60 probe/fc_s60_p2.jsonl --k 4 --items probe/leader_p2.jsonl
$PY cmp.py probe/leader_p2.jsonl probe/fc_s60_p2.jsonl
delta patch build --parent exp-leader/model --modified runs/fc/step60/model.safetensors --output submission
delta validate submission
```

Environment: RTX 3090 (24 GB), Python 3.12, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, `deltaleague` 0.3.0. About 4.5 GPU-hours for the 60 training steps, 25 minutes per probe, 15 minutes for the public suite. vLLM sampling on a shared GPU is not bit-reproducible, so a rerun will give different weights.

**sha256:**

- parent `model.safetensors`: `fc2a115f0507251baebef840d727b3bbf626ee861e8ed4d6bd33c0e7f81441e6`
- step 60 `model.safetensors`: `64e6b3833e2f5dd68733be1c272a4d20f295177b349f021da098ca8936ff25aa`
- patch `weights.safetensors` (sparse, 298 tensors): `cefeebf069287ac2025efb419df3cb88a6bf84de00ab0a1ed9b1e83d595137fd`
- `manifest.json`: `f8bc138c156381ca24de15ca2023f23fe3d4e32afa77427e82e2feadb71c3c9e`

`delta validate` passes.

## Limitations and negative findings

- One training run and one seed. There is no round-to-nearest run with the same pool, batch and step count, so the comparison with #81 is suggestive, not controlled: the pool, batch size and number of steps also changed.
- The probe pools are open training splits from the same families the run trained on. In #81 a pooled local gain of +0.95 [−0.06, +1.95] turned into −2.5 on the private test. This gain is larger and excludes zero on both sets, but the probe has not yet predicted a private result correctly.
- Code and function calling are not in the training or probe pools. On the public suite they moved by one to four items in either direction.
- Stochastic rounding changes 16% of the weights by one bf16 step each; how much of that is signal and how much is harmless noise was not separated.
- **Negative: three times the learning rate collapses.** The identical run at 3e-6 (same pool, seed and steps) kept normal answer length through step 40, then dropped abruptly from about 2,100 to about 1,200 tokens per answer. Its step 60 scores −5.58 [−7.62, −3.48] on probe set 1 (instruction following −11.6, math −6.4), with 30.8% of bf16 weights changed. With `--len-shape` rewarding brevity, unbiased export makes this collapse reachable; 1e-6 for 60 steps showed no sign of it (answers 3% shorter).
- Also measured and negative (no training): scaling the parent's final RMSNorm gain by 1.10 gives +0.40 [−0.90, +1.70] on probe set 1 and by 1.25 gives −0.88 [−2.23, +0.47], so further output sharpening of this parent does not help.
