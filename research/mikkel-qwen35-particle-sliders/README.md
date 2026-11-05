# Qwen3.5 particle teachers and mergeable reasoning sliders

This experiment compares a routed particle teacher, its ridge-distilled linear
student, and a supervised rank-8 LoRA on the registered original Qwen3.5-0.8B
checkpoint. It is local research; no official submission has been made.

The hypothesis is that the frozen model's hidden response when given a verified
worked solution can teach a neutral model to solve more accurately. The particle
route uses the same historical paired-error gmix formulation as
`mikkel/meta-glimmer-particle-slider`, with its source-checked upstream update.
An ordinary LoRA trained on the same source is a practical comparator.

Mikkel requested this experiment; Codex implemented and ran it. The agent could
access the registered public checkpoint, public suite manifest, public training
datasets, public CLI/source and leaderboard aggregates. It did not access hidden
questions or private generations. No official evaluation quota was consumed.

## Lineage and data

| Item | Immutable identity |
| --- | --- |
| Model | `qwen35-08b-v1`, `Qwen/Qwen3.5-0.8B` |
| Hugging Face revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Parent version | `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` |
| Parent canonical content hash | `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8` |
| Parent weights SHA256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Evaluation configuration | `public-ladder500-8k-v2` |
| Public suite manifest SHA256 | `996573d687b0d0f1e088f1e918cd153fa0f80cff454a7e9a7ab5ac2eb374d4d2` |
| GSM8K train source SHA256 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| Prepared data SHA256 | `66ed9d77ff1bc383d9f1a2259d75fd507cb75fbef17ed612aca67a8e63c1574d` |

Training uses only GSM8K's training split, downloaded from
`openai/grade-school-math` revision `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`.
Hash-sorting questions yields 128 training, 32 development and 32 separate ridge
calibration rows. The declared 384-token prompt/history cap retains 125 training,
32 development and 31 calibration rows. Public validation rows never become
gradient or ridge-fit data.

## Method and compute

All runs use seed 7, the full unquantized BF16 pretrained checkpoint, FP32 adapter
parameters, and genuine native 1024-wide final text hidden responses. Both hybrid
linear/full-attention and MLP projections are selected: 186 linear modules across
24 text layers. Vision, embeddings and output head remain unchanged. This is
adapter training through the full text model, rather than full-parameter SFT.

The particle adapter has one shared learned 128x4 bank; each projection has rank-8
down/up matrices and deterministic all-particle softmax routing. The upstream
gmix critic, paired-edit normalization, relativistic losses, lazy b_cap and
particle VIC remain unchanged. Positive and neutral contexts receive identical
continuation token IDs. There is no supervised auxiliary term in the GAN route.
The constant reference noise schedule remains 1600 updates even for shorter runs.

The exact historical source cohort is `HyperGAN/particle-sliders` revision
`1f5b380d317b73779a8aeab63b6bec0093c2c65c`, Glimmer integration
`80f22afd04d0794c1408c0058ec2b5f674a27b03`, and ParticleGAN
`82b72661c03377f348ccb76de37cd4ca02d45a8d` (package version 0.6.0).
This cohort is isolated from modern PG0.9.4, whose current Glimmer backend does
not admit this native hidden width. No modern admission gate is bypassed.

The finite search has four 32-update trials, batch four:

| Trial | G learning rate | D/G rate ratio | bank/G rate ratio | Distilled dev RMS / off RMS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.0001 | 1.5 | 10 | 0.993258 |
| 1 | 0.0003 | 1.5 | 10 | 0.985407 |
| 2 | 0.0006 | 1.5 | 10 | 0.981180 |
| 3 | 0.0003 | 0.75 | 2.5 | 0.993702 |

Trial 2 was selected for a fresh 256-update final run, completed in 773.1 seconds.
The selected EMA teacher has dev normalized hidden RMS 1.026729, versus off
1.105955 and raw teacher 1.134817. Its distilled student has RMS 1.028029.
This selection diagnostic
measures held-out hidden-state error, not language capability. A separate
256-row-per-projection calibration calls the unchanged upstream dual ridge solve
with ridge fraction 0.01, retaining each teacher up matrix. The resulting linear
student is an approximation, evaluated separately from the nonlinear teacher.

The supervised comparator uses ordinary rank-8 LoRA on the same modules, answer
token cross entropy, AdamW at 0.0001, batch four, and 128 updates. Its held-out
train-source token NLL is 0.618644. Two local RTX A6000 48GB cards were used;
search runs sequentially on one card while the comparator/evaluation uses the
other. No rented compute or seed search was used.

Runtime: Python 3.12.15, PyTorch 2.15.0+cu132, Transformers 5.22.1, safetensors
0.8.0, lm-eval 0.4.13, Delta CLI 0.1.1. Native PyTorch reference linear-attention
kernels were used; no flash-linear-attention or causal-conv1d package was added.

## Public advisory results

The current CLI's `delta evaluate` expects benchmark-to-example lists, whereas
the live hashed download is a newer task-selection manifest. The experiment
preserves that file and executes the manifest-pinned lm-eval likelihood tasks.
Every selected document hash, population, index and dataset revision is verified.
The likelihood subset uses no chat template, `acc_norm` where available and
`acc` otherwise, 64 examples per task, fixed lm-eval seeds 0/1234 and max context
2048. These eight tasks are 512 public development examples, not the full suite.

| Benchmark | Parent | Supervised LoRA |
| --- | ---: | ---: |
| ARC Challenge | 32.8125 | 39.0625 |
| ARC Easy | 62.5000 | 70.3125 |
| BoolQ | 65.6250 | 73.4375 |
| CommonsenseQA | 57.8125 | 54.6875 |
| HellaSwag | 48.4375 | 51.5625 |
| OpenBookQA | 29.6875 | 34.3750 |
| PIQA | 70.3125 | 70.3125 |
| WinoGrande | 56.2500 | 54.6875 |
| Equal-weight subset mean | 52.9297 | 56.0547 |

The official CLI's `delta compare` gives +3.125 points, a paired bootstrap 95%
interval of approximately +0.6 to +5.7 points (10,000 resamples, seed 0), and a
pooled sign-test p-value of 0.023. This describes the selected local protocol,
not a claim about private-test performance. CommonsenseQA and WinoGrande regress.

Generated math uses the first 16 manifest-selected GSM8K test examples, seed 1000,
thinking enabled, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5,
and a deliberately bounded 1024-token output cap versus the registered 8192.
These are zero-shot user-question chat prompts; the native suite's GSM8K task
declares five few-shot examples. The LoRA answers 2/16 correctly, with no
truncations; the parent scores 4/16 and truncates every generation. The paired
difference is -12.5 points with interval [-37.5, +12.5], sign-test p=0.625, so
it is not separable at this sample size. This screen cannot establish official
math performance.

All 114 manifest-selected MMLU examples were then added. Full-strength LoRA
scores 45.6140 versus parent 47.3684 on MMLU. Combining the disjoint likelihood
reports with `combine.py` gives nine-task means parent 52.3118 and LoRA 54.8946.
The paired interval is [0.2284, 4.9373], but pooled sign-test p=0.0925, so the
CLI's joint separability criterion is not satisfied.

A bounded second strength, 0.5, scores 53.3534 on the nine-task likelihood mean,
preserves MMLU at 47.3684, and answers 6/16 math questions with no truncations.
Its likelihood difference is +1.0417 (interval approximately [-0.6, +2.7]);
the math difference is +12.5 (interval [-18.75, +43.75]). Both are inconclusive.
We prefer this balanced supervised candidate for a first submission, while
retaining full strength and the particle artifacts as alternatives. This is
exploratory development-set selection, not an independent confirmation.

The final particle student scores 50.3807 on the same nine-task likelihood
selection, versus parent 52.3118: -1.9310 points, interval approximately
[-4.1, +0.2]. The hidden-state improvement did not transfer to a positive
capability signal in this comparison. Native bank probes nevertheless confirm
that its reloaded teacher has exact OFF behavior and a live 128x4 bank: zeroing
the bank changes hidden responses by RMS 0.137157 on three generic probes.
The teacher's total ON/OFF hidden RMS is 3.213307 on those probes. These are
mechanics checks, not quality scores. Final generated particle screens are
still in progress and will be recorded before this PR leaves draft.

## Reproduction and artifacts

Implementation source: [HyperGAN/qwen35-particle-sliders at
45ec4a4](https://github.com/HyperGAN/qwen35-particle-sliders/tree/45ec4a4fb44bd450b087f4e1926adaca2068b62d).
The companion `reproduce.py` prepares portable local configurations and checks
source-data identity; the project README lists the training/evaluation commands.

```sh
git clone https://github.com/HyperGAN/qwen35-particle-sliders.git ../qwen35-particle-sliders
git -C ../qwen35-particle-sliders checkout 45ec4a4fb44bd450b087f4e1926adaca2068b62d
uv venv --python 3.12.15 ../qwen35-venv
uv pip install --python ../qwen35-venv/bin/python 'torch==2.15.0' -e '../qwen35-particle-sliders[test,eval]'
. ../qwen35-venv/bin/activate
python research/mikkel-qwen35-particle-sliders/reproduce.py --code ../qwen35-particle-sliders --output ../qwen35-repro
delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output ../qwen35-repro/experiment
qwen35-slider search ../qwen35-repro/search.json
qwen35-slider train-lora ../qwen35-repro/lora.json
qwen35-slider build --workspace ../qwen35-repro/experiment --deltas ../qwen35-repro/runs/lora-v1/deltas.safetensors --strength 0.5 --output ../qwen35-repro/submission
delta validate ../qwen35-repro/submission
qwen35-slider verify-candidate --workspace ../qwen35-repro/experiment --deltas ../qwen35-repro/runs/lora-v1/deltas.safetensors --strength 0.5 --patch ../qwen35-repro/submission
```

The regenerated workspace must retain the evaluation ID and hashes above.
Expected artifact hashes identify the recorded run; a different CUDA stack may
produce different regenerated bytes, so rebuild and verify the manifest rather
than reusing an old hash. `particle_probe.py` reproduces the native bank check.
The product's source, wheel and sdist module bytes match, and both installed
CPU integration tests pass. Native checks are separate from those fixtures.

The LoRA dense replacement patch has passed `delta validate`, complete tensor
reconstruction, registered canonical parent identity verification, and bitwise
loaded logits versus the evaluated direct-delta model on three serialization
probes. Probe equality is a mechanics check, not a quality evaluation.

| LoRA artifact identity | SHA256 / canonical content hash |
| --- | --- |
| Projection deltas | `68b6dfa452168cbc690405af817341a0533bc47439f91df8c482cdc956b1bf39` |
| Submission weights | `4e142fcd549a1fd9046245cd8f0e11048ef8c48ebb9ff853afc619e72a19c2b3` |
| Reconstructed checkpoint file | `6ae5f7610be1c986614daebdcd48f1e5c143f1a501b0f74f7e22579e47afb2b7` |
| Result canonical model content | `a239ac71468fc7402b210ebd006d2dc15ad7e1d6adb69131363823d25fa833c8` |

Preferred strength-0.5 patch: weights SHA256
`7c19870c30cfb09587ea0c26c2f9d27537dfdc9ea6281da297d0197682a1ec84`,
result canonical content
`b2dfb0fb11e30c606f4366b81b29487ca017745ce0bdc5da49cfc589a0a318e4`.
The particle patch's weights SHA256 is
`3927c5dcbf9a7b6855ccd719a625dbc7553f113ce218db94ed1b4a1b4190f2e2`,
result canonical content
`c93e2237b3cde2e9921025f801799913c1a8f0518a83cdfda4364eddda38b538`.
All three patches passed exact tensor reconstruction and loaded-logit checks.

Weights, private material, credentials and raw generation logs are excluded from
this PR. Aggregate evidence and metadata are suitable for public release.

## Limitations and submission

The training source is small and math-focused. Hidden-state matching and a small
calibration set may not transfer to generation. The local subset omits MATH500,
BBH, MMLU, MMLU-Pro, IFEval, coding and BFCL; public-development selection can
overfit. The registered private score and complete validation score remain
unknown. No score is inferred from successful serialization or fixture tests.

The preferred artifact is the supervised strength-0.5 patch; the particle teacher,
distilled student and valid particle patch are retained as experimental results.
No official submission ID or URL exists; an actual quota-consuming submission
is a separate account action after this research PR is opened.
