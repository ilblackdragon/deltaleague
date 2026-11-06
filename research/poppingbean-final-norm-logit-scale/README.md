# Final-norm logit scale (α = 1.25) for Qwen3.5-0.8B

A single-tensor, data-free edit: multiply the final RMSNorm gain of the language
model by α = 1.25. Embeddings are tied, so this multiplies every output logit by
1.25. Under the pinned private-test decoding (temperature 1.0), that is the same as
sampling at temperature 0.8.

## Hypothesis and scope

- **Hypothesis.** `public-ladder500-8k-128-v1` samples private-test answers at
  T = 1.0, top_p 0.95, top_k 20, presence penalty 1.5, thinking on, up to 8,192
  tokens. For a 0.8B model, sampling at T = 1.0 may add avoidable errors to long
  reasoning traces. Scaling logits by α is the same as sampling at T/α = 0.8, and
  the logit-space presence penalty becomes 1.5/α = 1.2. If sharper sampling helps
  more than it hurts, the private score rises.
- **Why public validation should barely move.** Generative public tasks decode
  greedily, and greedy argmax does not change when logits are scaled. Likelihood
  tasks whose answers are a single token keep their ranking. Only multi-token
  likelihood comparisons, such as acc_norm over endings, can change.
- **Main risk (stated in advance).** A lower temperature can make the repetition
  loops that small thinking models fall into worse, which costs more answers at
  the 8,192-token cap.
- **Not a search.** α = 1.25 was chosen once, before any measurement, as a
  conservative step toward Qwen's older thinking-mode temperature of 0.6. No
  other α was evaluated.
- **Attribution and access.** The edit was proposed by an AI agent (Claude Code)
  and approved by the human contributor (`poppingbean`), who also set the compute
  budget. The research used only public material: this repository, the public
  checkpoint, `evaluation.json`, `validation.json`, and public Hugging Face
  datasets at the pinned revisions. There was no access to private tests.

## Lineage and configuration

| Item | Value |
| --- | --- |
| Model / competition | `qwen35-08b-v1` |
| Parent version | `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` (original model) |
| Parent content hash | `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8` |
| Parent `model.safetensors` sha256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Evaluation config | `public-ladder500-8k-128-v1` (`evaluation.json` sha256 `4c72581d…c11f1`) |
| Public suite | `qwen-small-public-v1`, `validation.json` sha256 `996573d6…74d4d2`, seed 20261104 |
| CLI | `deltaleague` 0.1.2 at `ilblackdragon/deltaleague@1826942` |

## The edit

`model.language_model.norm.weight` (BF16, shape [1024]) is the only tensor changed.
Qwen3.5 RMSNorm computes `x̂ · (1 + w)`, so `w' = bf16(1.25 · (1 + w) − 1)`.

| Check | Result |
| --- | --- |
| Changed entries | 1,024 / 1,024 (one tensor) |
| Original tensor sha256 (raw BF16 bytes) | `30832e6642a9c1555485d5c9553a35e539278c33ce3b7a9665e06cd2c5598381` |
| Edited tensor sha256 (raw BF16 bytes) | `6960071031ac175ce8e1b8efe37ef4abd5b24e970666ee8e49473a6cf19c1dcb` |
| Realized per-channel scale after BF16 rounding | mean 1.25001, min 1.24561, max 1.25373, sd 0.0019 |
| Patch | sparse-diff / replace, `weights.safetensors` sha256 `4960af4903d3199c17fef126c6c50b9754ec9aa3d04246f4f577f29c7bc0ce08` (10,400 bytes) |
| Patch manifest sha256 | `6e98760493e5eee70d01601c859fe942ce2fd1c178cdfc4c1b57bb7b2a7cf71c` |
| Determinism | Rebuilding the patch gives byte-identical files; applying it to the parent reproduces the edited tensor bit for bit |

Logit check (`results/logit-check.json`, CPU FP32, 3 prompts): the least-squares
fit of edited logits against original logits gives a slope of 1.2496–1.2510, with
relative residual 0.1–0.25%. Greedy argmax agrees at 100% of positions.

## Reproduction

```sh
pip install deltaleague            # 0.1.2 or this repo at 1826942
delta competition init qwen35-08b-v1 --output experiment
cd experiment
python ../research/poppingbean-final-norm-logit-scale/make_edit.py model/model.safetensors edited.safetensors
delta patch build --parent ./model --modified ./edited.safetensors --output ./submission
delta validate ./submission
```

The edit needs no data and no seed. The local checks below also need
`lm-eval==0.4.13`, `accelerate`, and the repo's `delta_cli.public_eval` (install
this repo with `pip install --no-deps .`).

```sh
# make a model dir for the candidate: copy model/ minus weights, add edited.safetensors as model.safetensors
python check_logits.py WORKSPACE
python likelihood_eval.py WORKSPACE WORKSPACE/model base.json --device cuda --batch 16
python likelihood_eval.py WORKSPACE WORKSPACE/edited-model edit.json --device cuda --batch 16
python compare_items.py base.json edit.json            # paired bootstrap, 10,000 resamples, seed 0
python gsm8k_sampled.py WORKSPACE WORKSPACE/model WORKSPACE/edited-model gsm8k.json gsm8k.generations.jsonl \
       --n 40 --max-tokens 4096 --deadline-minutes 95 --device cuda
python rescore_gsm8k.py WORKSPACE gsm8k.generations.jsonl gsm8k-rescored
python compare_items.py gsm8k-rescored.base.json gsm8k-rescored.edit.json
```

**Hardware and software used.** Windows 11 with an AMD Ryzen AI Max+ 395 and its
integrated Radeon 8060S (gfx1151). The GPU ran PyTorch 2.9.1+rocm7.13.0 from AMD's
gfx1151 wheel index, with Transformers 5.22.1 and lm-eval 0.4.13, in BF16. The
fast linear-attention kernels (flash-linear-attention, causal-conv1d) are
unavailable on this platform, so Transformers' reference implementations ran.
There was no vLLM and no rented compute. The agreed budget was ≤ 2.5 h of local
compute; about 2 h 20 min was used: about 10 min of CPU checks, 27 min for the
likelihood runs, and 102 min for the gsm8k run.

## Evidence (all local and advisory; none of these are official scores)

### 1. Public log-likelihood benchmarks (9 of 18)

These are the exact pinned public items, loaded through `delta_cli.public_eval.text`
(pinned revisions, document-hash verification) and scored with lm-eval 0.4.13's
HuggingFace backend instead of the vLLM completions endpoint. Primary metrics
follow `delta_cli.public_eval.suite.PRIMARY`. GPU BF16, batch 16. Files:
`results/ll-base.json`, `results/ll-edit.json`, `results/ll-compare.json`.

| Benchmark | n | Metric | Parent | α = 1.25 | Diff | Wins / losses |
| --- | --: | --- | --: | --: | --: | --: |
| arc_challenge | 64 | acc_norm | 31.25 | 28.12 | −3.12 | 1 / 3 |
| arc_easy | 64 | acc_norm | 64.06 | 65.62 | +1.56 | 1 / 0 |
| boolq | 64 | acc | 65.62 | 65.62 | 0.00 | 0 / 0 |
| commonsense_qa | 64 | acc | 56.25 | 57.81 | +1.56 | 1 / 0 |
| hellaswag | 64 | acc_norm | 48.44 | 45.31 | −3.12 | 1 / 3 |
| mmlu | 114 | acc | 46.49 | 48.25 | +1.75 | 2 / 0 |
| openbookqa | 64 | acc_norm | 29.69 | 28.12 | −1.56 | 0 / 1 |
| piqa | 64 | acc_norm | 71.88 | 73.44 | +1.56 | 1 / 0 |
| winogrande | 64 | acc | 57.81 | 60.94 | +3.12 | 2 / 0 |
| **Equal-weight mean** | 666 | | **52.39** | **52.58** | **+0.19** | 9 / 7 |

The paired bootstrap 95% CI for the mean difference is [−1.04, +1.49], and the
pooled sign test gives p = 0.80. **No detectable change, as predicted.** mmlu
answers are single tokens, so its 2 flips can only come from BF16 rounding near
ties. That gives a sense of the noise floor. The parent's arc_challenge score
differs by one item between CPU FP32 (32.81) and GPU BF16 (31.25), another reminder
that these numbers are advisory.

The other 9 public benchmarks are greedy and generative (gsm8k, math500, bbh,
mmlu_pro, ifeval, humaneval, mbpp, bfcl×2). Argmax doesn't change under logit
scaling, so they should be unaffected apart from BF16 tie-breaking. They were
**not** run locally, because vLLM is unavailable on this platform.

### 2. Private-test-like regime: gsm8k, thinking on, sampled

Setup:
- **Items:** the first 40 public-validation gsm8k items in ascending index order,
  hash-verified. Each item ran on both models with seed 1000 + position.
- **Sampling:** the CLI's own `Runner` with `evaluation.json`'s settings (T 1.0,
  top_p 0.95, top_k 20, presence 1.5, thinking on), except **max_tokens 4,096**,
  half the private cap, to fit the compute budget.
- **Prompt:** the question followed by "Please reason step by step, and put your
  final answer within \boxed{}." The private prompt format is unknown.
- **Stopping rule:** no new item after 95 minutes. **18 paired items** finished.

Files: `results/gsm8k-rescored*.json` and `results/gsm8k-compare*.json`. Raw
generations were kept locally and are deliberately not committed.

| Score (n = 18 paired) | Parent | α = 1.25 | Diff | Wins / losses | Sign test p | Paired bootstrap 95% CI |
| --- | --: | --: | --: | --: | --: | --: |
| Lenient: CLI rule (text after `</think>`; the whole text if the trace never closed) | 27.8 | 50.0 | +22.2 | 5 / 1 | 0.22 | [0.0, +44.4] |
| Strict: only answers after a closed thinking trace | 5.6 | 11.1 | +5.6 | 1 / 0 | 1.00 | [0.0, +16.7] |
| Truncated at 4,096 tokens | 16 / 18 | 16 / 18 | 0 | | | |
| Mean generated tokens | 4,036 | 3,984 | | | | |

How to read this:
- Both models usually fail to finish thinking within 4,096 tokens. The traces
  fall into "wait, let me re-read" loops; this is the base model's behaviour at
  these settings.
- **Most lenient credit comes from unfinished traces**, where math_verify happens
  to pick up the correct number from the reasoning. The direction favours the
  edit, but that does not show the edit produces better final answers.
- The strict difference is a single item.
- No difference in truncation was observed, so the main risk stated in advance
  (more looping at the lower temperature) did not show up at this sample size.
  That is not evidence that it is absent.
- **This is a weak, non-significant directional signal and should not be read
  as a measured improvement.** The private-test score is the real test.

## Limitations and negative findings

- No local result can measure the claimed private-test effect with useful
  precision: the sample is small, the token cap is half the real one, and the
  numerics are Transformers BF16 rather than vLLM. Only the server's private-test
  score can confirm or refute the hypothesis.
- The pinned evaluator itself (`delta public-eval run`) could not be run, because
  it needs vLLM on a Linux GPU. The likelihood numbers use the same tasks, items
  and metrics through a different backend.
- **Tooling issue found (not fixed here).** In this Windows/Git Bash environment,
  `delta_cli.scoring.score_answer(..., scorer='math')` returned 0.0 even for
  correct answers. math_verify's Windows timeout uses multiprocessing, and its
  child processes fail with `OSError: [WinError 6] The handle is invalid`. The
  gsm8k numbers above therefore use `rescore_gsm8k.py`, which makes the same
  math_verify calls in-process with timeouts disabled. This should go in a
  separate tooling PR or issue.
- Changing the effective temperature affects every task sampled at T > 0. It does
  not make the model know anything more.

## Submission

Not yet submitted. Pending.
