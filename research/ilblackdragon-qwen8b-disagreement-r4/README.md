# Qwen3-8B disagreement corrections for Qwen3.5-0.8B

## Hypothesis and scope

Illia Polosukhin proposed using a larger Qwen model's disagreements to improve the
smaller competition model. Codex implemented and ran one local experiment:
learn a rank-4 update to the final MLP down-projection using examples where
Qwen3-8B's answer is correct and the original student's answer is wrong.
Correctness is checked against public labels; disagreement alone is insufficient.
The prediction is that this targeted edit improves a disjoint public holdout.

Only the public repository, public competition metadata/checkpoint/validation
manifest, public ARC-Challenge data, and public teacher weights were available.
No hidden evaluation examples were accessed. Existing original leaderboard scores
were read for later comparison. There was one fixed parameter setting, no search.

## Lineage

- Model: `qwen35-08b-v1`.
- Parent: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` (Original model).
- Parent content hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Evaluation: `public-ladder500-8k-v2`.
- Public validation file SHA-256: `996573d687b0d0f1e088f1e918cd153fa0f80cff454a7e9a7ab5ac2eb374d4d2`.
- Teacher: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- ARC data: `allenai/ai2_arc`, `ARC-Challenge`, public `test` split,
  revision `210d026faf9955653af8916fad021475a3f00453`.
- Repository examined: `d5fb6169fe6384c20077811c4860fb59874bd5b8`.

The upstream ARC split named `test` is public development material selected by
the League's public manifest; it is not the League private test.

## Method

The manifest selects 64 ARC-Challenge documents. All 64 canonical UTF-8 document
hashes are checked before inference. Python `random.Random(42).shuffle` creates
32 development and 32 holdout indices. The teacher sees only the development
half. IDs and resulting aggregates are recorded in `results.json`.

Each model receives the same question and letter-labeled choices using its own
chat template, with thinking disabled. We take the argmax of the next-token
logits for valid choice letters. No tokens are generated, no sampling is used,
and the prompt limit is 512 tokens. Both models run in float32 on CPU.
This deliberately cheap answer-letter proxy differs from the pinned native
ARC `acc`/`acc_norm` metric, and is **not** an official validation score.

The editable tensor is `model.language_model.layers.23.mlp.down_proj.weight`.
All other parameters are frozen. Cache the final projection inputs and pre-final-
normalization residuals, and fit `B @ A` (rank 4, alpha 4). A starts from
Normal(0, 0.01), B from zero, with torch seed 42. Forty full-development-batch
AdamW steps use learning rate 0.001, zero weight decay, and gradient norm clip 1.
The loss is answer-choice cross entropy on verified teacher corrections plus
0.5 times KL divergence from the original student's choice distribution on all
development examples. Holdout examples are excluded from all optimizer losses.

After training, scale the update down if needed to cap its Frobenius norm at
0.5% of the parent tensor norm, add in float32, and cast to the parent's BF16.
The rounded checkpoint difference need not have exact rank 4. Save the full
checkpoint separately as `edited.safetensors`. No candidate is selected or
retuned using holdout scores.

Cached calculations must reproduce the original logits. A fresh model forward
with the rounded candidate must reproduce cached candidate logits. `audit.py`
checks original file hashes, finite tensors, matching shapes and dtypes,
one-tensor edit scope, and bitwise equality between patch reconstruction and
the exported checkpoint.

## Reproduction

Python 3.12, Linux CPU. Dependencies are pinned in `requirements.lock.txt`.
Install the CPU PyTorch wheel from its official index, then the lock file.
The user's approved limit was two hours, 16 CPU threads, 64 GiB RAM, and 25 GiB
additional disk, with no rented compute. `reproduce.py` gives download, inference,
training, patch build and validation a combined 7200-second deadline. Inference
processes also enforce a 64 GiB address-space limit. Run stages sequentially.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r research/ilblackdragon-qwen8b-disagreement-r4/requirements.lock.txt
delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output experiment
python research/ilblackdragon-qwen8b-disagreement-r4/reproduce.py --workspace experiment
python research/ilblackdragon-qwen8b-disagreement-r4/audit.py --workspace experiment
```

Run from a checkout containing this experiment. Use a fresh workspace for
reproduction: the script refuses to overwrite caches/candidates. Initialization
must still pin `public-ladder500-8k-v2` and the public validation hash above;
the service's default evaluation may change. Exact hashes may depend on the
recorded software/hardware; the recorded run used a Ryzen 9 9950X3D.

## Research trace and limitations

1. Read README, ARTIFACTS, AGENTS, CONTRIBUTING and LOCAL. Installed PyPI CLI
   0.1.1 in a local virtual environment after system Python rejected global pip.
2. Completed browser login; listed the competition; downloaded the parent without
   modifying it. Quota before research was 0/5 used, 36,000 GPU-seconds daily.
3. The public file is a newer selection manifest. CLI 0.1.1 `delta evaluate`
   rejects it with `Public validation benchmark inventory mismatch.` before
   inference. Neither the downloaded data nor shared scoring code was changed.
   No CLI report or `delta compare` result is fabricated for this proxy.
4. Read only the public ARC selections from the pinned upstream dataset. An
   initial hash check failed for the one non-ASCII document because JSON's
   default ASCII escaping was used. Corrected serialization to UTF-8
   (`ensure_ascii=False`); all 64 hashes then matched before model inference.
5. Ran a single baseline, development teacher pass, fixed adapter fit, and
   candidate pass. No long generations or paid inference were used.

The 32-example public holdout is small and is not an untouched private test.
Development scores are training-contaminated. Paired bootstrap intervals use
10,000 resamples, seed 42; exact two-sided sign-test p-values accompany changes.
A gain on this proxy does not establish broad model improvement or predict
official native-metric performance. A negative result is retained without tuning.

## Local findings

The teacher scored 29/32 (90.625%) on development and supplied 11 verified
corrections. This one update produced the following local answer-letter results:

| Public split | Original | Candidate | Difference | Paired wins/losses | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development (used in training) | 18/32 (56.25%) | 20/32 (62.5%) | +6.25 pp | 2 / 0 | 0.5 |
| Holdout | 14/32 (43.75%) | 14/32 (43.75%) | 0 pp | 0 / 0 | 1.0 |

**No holdout improvement was demonstrated.** The development bootstrap interval
is [0, 15.625] percentage points. The holdout empirical paired bootstrap interval
collapses to [0, 0] because all observed correctness outcomes tie; this is not
evidence of zero population uncertainty. The small public sample and changed
metric limit any conclusion about official scores. The experiment does not
justify claiming that the larger model improved general student capability.

The unrounded update was scaled by 0.08593208586290096 to the fixed 0.5% norm
limit. BF16 rounding makes the actual saved relative difference 0.5161004%.
Exactly 2,401,494 elements in one tensor changed. All seven original downloaded
files passed their hash checks; all 488 checkpoint tensors retained names, shapes
and dtypes; patch reconstruction equals the edited checkpoint bit for bit.
`delta validate` passed. The sparse replacement patch is 24,015,116 bytes.

Student baseline, teacher scoring, and edit/candidate verification took 27.34,
97.41, and 28.86 seconds respectively (153.60 seconds combined). Peak measured
RSS was 46.20 GiB for the teacher. Download/setup/patch auditing are additional
wall time, still within the approved two hours. No GPU or paid compute was used.
The workspace occupies about 19 GiB including the original, teacher, edit and
patch; the virtual environment occupies about 1.5 GiB.

Expected full edited checkpoint SHA-256:
`92c44142e53262067e349f3fd8374b960e6479eb1dde3c3da3611f4d275c91e9`.
Patch SHA-256:
`baf46a93851e7f22b856cba246ac677368cb2ae090acd2e017a0d1d5e3ae6d94`.
Full aggregate evidence, tensor hashes, and timings are in `results.json`,
`audit.json`, and `runtime.json`; no weight artifacts or raw generations are
included in this research directory.

## Verified baseline and submission

The existing original result on `public-ladder500-8k-v2` is **41.830071900430084
public validation / 18.6 private test**, completed 2026-11-04. The evaluation
metadata marks the private test provisional. These scores are platform-verified;
they are not the local answer-letter proxy.

Candidate submission: **not submitted; awaiting explicit quota approval**.
Candidate official public/private scores: **unavailable**. A submission URL and
verified comparison will be added only after an approved evaluation completes.
