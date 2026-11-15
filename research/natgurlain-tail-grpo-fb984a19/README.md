# Tail-only GRPO on fb984a19: train only on prompts the parent gets wrong sometimes or answers at length

Research by natgurlain with an AI agent (Claude Code). Available during research: public leaderboard rows, public checkpoints, public research PRs, public training splits of open datasets and the `delta` CLI. No private test material.

- **Parent:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, rank 1, private 53.4), content hash `bd3c11a04a4d1691c1de6b7afadd18d8a56c232d29aa8959fec41819ff43ddfd`
- **Evaluation configuration:** `private1000-v6-thinking8k-20261111` (`delta-public-v1`)
- **Submission:** `bb069288-ff60-42de-bd95-36bcb0c16744` (version `856bfe07-2ed5-4148-97a3-5e15ec0114ea`), verified: private 50.9 ± 1.58 against the parent's 53.4

All statements about quality below are local and advisory unless marked verified.

## Hypothesis

GRPO on this parent (#58, #74, #77) draws prompts uniformly from the training pools. Most of those prompts carry no gradient: the parent either answers all samples correctly and briefly, or fails all of them. If training batches are restricted to prompts where the parent's samples disagree or run long, each step should move the weights further in the useful direction, and a short run should lower truncation and raise accuracy together.

## Method

1. **Mine.** `mine.py` samples the parent 4 times on 3,000 training-split prompts (math 35%, multiple choice 30%, instruction following 25%, logic 10%) with the evaluation sampler (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 8,192 tokens).
2. **Select.** `select_tail.py` keeps a prompt if its 4 samples disagree, or if any sample reaches 4,096 tokens. 1,330 of 3,000 are kept (761 mixed, 321 long, 248 both); see `results/tail_stats.json`.
3. **Train.** `grpo.py` (the #58 recipe with the #74 length shaping) for 16 steps of 128 prompts x 8 samples drawn from the kept prompts only: learning rate 1e-6, 6,144-token training cap, `--match-sampler`, `--len-shape 3000`, seed 20261115. Exact commands: `night_b.sh` and `night_c.sh` (the run was paused after step 9 and resumed from saved full-precision state).
4. **Probe.** Each checkpoint and the parent are sampled 4 times on 1,000 held-out training-split prompts that were excluded from mining, with the same request stream, and compared per prompt (`cmp.py`).

One run, one seed. Checkpoints were saved at steps 4, 8, 12 and 16; steps 8, 12 and 16 were probed.

## Evidence

Held-out probe, 1,000 prompts x 4 samples, parent 65.33:

| Checkpoint | Score | Difference | 95% interval | Truncated at 8,192 | Mean tokens |
|---|---:|---:|---|---:|---:|
| parent | 65.33 | | | 7.15% | 2,028 |
| step 8 | 65.20 | −0.12 | [−1.35, +1.12] | 6.33% | 1,930 |
| step 12 | 64.22 | −1.10 | [−2.45, +0.25] | 5.22% | 1,764 |
| step 16 | 66.40 | +1.07 | [−0.25, +2.50] | 4.20% | 1,634 |

Step 16 was then re-tested on a second set of 1,000 fresh prompts (excluded from mining and from the first probe; `night_d.sh`):

| | Parent | Step 16 | Difference | 95% interval | Truncated | Mean tokens |
|---|---:|---:|---:|---|---:|---:|
| second set | 65.58 | 66.40 | +0.83 | [−0.60, +2.25] | 6.38% → 3.27% | 1,928 → 1,522 |
| both sets pooled (n = 2,000) | | | +0.95 | [−0.06, +1.95] | | |

- **Length and truncation:** fall steadily with training. At step 16 truncation is roughly halved and answers are about 20% shorter. This part of the hypothesis is supported locally.
- **Accuracy:** not established. No single comparison excludes zero. The second set is the only one free of selection (step 16 was chosen after seeing the first), and its interval is [−0.60, +2.25]. Step 12 of the same run measured −1.10, so the path is not monotonic.
- **By pool on the second set:** choice +0.8 ± 1.4, instruction following +0.5 ± 1.6, logic +1.8 ± 2.6, math +0.9 ± 1.0.
- **Training reward** (`results/train_log.jsonl`) is flat within noise: about 0.47 for steps 1–4, 0.42 for steps 5–9, 0.47 for steps 10–16. Each step draws different prompts.

The full pinned public suite was not run for this candidate. Earlier PRs on this parent (#69, #74, #77) report that local probes did not predict the private ordering at this noise level, and the private test's standard error is 1.58.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output exp-leader
ln -s ../exp-leader/model parent        # inside the directory holding these scripts
./night_b.sh                            # mine, select, train (probe/leader.jsonl must exist: see probe() in gpu_lib.sh)
./night_c.sh                            # resume to step 16 and probe, if interrupted
./night_d.sh                            # second probe set
delta patch build --parent exp-leader/model --modified runs/tail/step16/model.safetensors --output submission
delta validate submission
```

Environment: RTX 3090 (24 GB), Python 3.12, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, `deltaleague` 0.3.0. Roughly 6 GPU-hours for mining and training and 20 minutes per probe. vLLM sampling on a shared GPU is not bit-reproducible, so a rerun will give different weights.

**sha256:**

- parent `model.safetensors`: `fc2a115f0507251baebef840d727b3bbf626ee861e8ed4d6bd33c0e7f81441e6`
- step 16 `model.safetensors`: `8b92d4042545daad17e458226d967b2d13410cf57046613925b32c49cd5362df`
- patch `weights.safetensors` (sparse, 264 tensors): `8dce3ebfa7a111acbc8d67a0efdfd20232fd74926d83e02efceecc3f4340d811`
- `manifest.json`: `dc79fa9f55551f0b86b9a829362a153287546ccd753fdbfd25b36db86a9cee19`

`delta validate` passes.

## Verified result

Official, from `delta submission status` on `private1000-v6-thinking8k-20261111`:

| | Private test | Private suite passed (of 500) | Validation | Truncation | Output tokens |
|---|---:|---:|---:|---:|---:|
| Parent `fb984a19` | 53.4 | 279 | 47.40 | 14.2% | 1,324,788 |
| **This submission `856bfe07`** | **50.9 ± 1.58** | **263** | **46.91** | **7.0%** | **1,042,855** |

- **The accuracy half of the hypothesis is not supported.** The private score is 2.5 below the parent (rank 18 at submission time). With a standard error of 1.58 per row this does not establish a loss, but it rules out the gain the local probes suggested.
- **The length half held:** truncation on the private test halved (14.2% → 7.0%) and output tokens fell 21%, as measured locally.
- **Negative findings:** (1) halving truncation at this level bought no score, so the answers that still run out of tokens on this parent are probably ones it would get wrong anyway; (2) a +0.95 [−0.06, +1.95] paired gain on 2,000 held-out training-split prompts did not carry over to the private test, in line with what #69, #74 and #77 reported for local probes.

## Limitations

- One training run and one seed; the checkpoint was picked among three by the first probe.
- The probe pools are open training splits chosen to resemble the public suite; they are not the private test.
- Code and function-calling tasks are not in the training or probe pools, so any effect on them is unmeasured.
