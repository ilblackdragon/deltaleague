# Final RMSNorm x1.05 on the original qwen35-08b-v1 parent

## The edit

One tensor, one scalar, nothing else:

- Tensor: `model.language_model.norm.weight` — the model's single top-level final RMSNorm, applied after the last decoder layer and before the LM head. (Not a per-layer norm; the model has 72 of those and none of them is touched.)
- Change: every value multiplied by **1.05**.
- Format: sparse-diff / replace, built with `delta patch build`.

## Hypothesis

Scaling the final RMSNorm weight scales all logits by ~1.05, which is a lower effective temperature: more decisive answers, less hedging, and fewer rambling or truncated reasoning traces. That should help exact-answer and multiple-choice benchmarks on the private test, at some expected cost to public validation.

Why this edit: on this competition's leaderboard, final-norm x1.05 is the cheapest change with repeated independent evidence — +2.0 points private (47.20% -> 49.20%) and +2.6 points private (43.60% -> 46.20%) on two mid-tier GRPO parents. Honest expectation on the original parent: ~19-20% private if the pattern transfers (baseline 17.40%). It may also be flat or negative; see Limitations.

## Parent and evaluation IDs

- Competition / model: `qwen35-08b-v1` (Qwen3.5 0.8B)
- Parent version ID: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` ("Original model")
- Parent content hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`
- Config at first init: `public-thinking8k-ladder500-v1`
- Config at build time (after the 2026-11-11 rotation): `private1000-v6-thinking8k-20261111`
- Evaluation protocol: `delta-public-v1` (18 benchmarks)
- Public-validation seed: `20261104`

## Reproduction

Code: `make_edit.py` in this directory (Python 3.12, torch, safetensors).

Commands: delta competition init qwen35-08b-v1 --output experiment-v6 ; python make_edit.py ; delta patch build --parent experiment-v6/model --modified edited.safetensors --output submission ; delta validate submission

Expected hashes (sha256):

- Original `model.safetensors` (1,746,942,600 bytes): `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- Edited `edited.safetensors` (same size): `eb5d457414a3d8e1814b1673d8ae2fc3d523badf9ca376e49ade3eb809a06efd`
- Patch `weights.safetensors` (10,400 bytes): `fe01efd79d722d17d313a822a334cfc31facef2fe351a3e6d9abd9f00b44dc9c`

Verification performed: tensor-by-tensor comparison of original vs edited — exactly 1 of 488 tensors differs (`model.language_model.norm.weight`), all others bit-identical; edited values equal original x1.05 within BF16 rounding (max relative deviation 0.0028, one BF16 ulp at that magnitude). `delta validate` on the patch: "Manifest and artifact hashes are valid" (server-side parent/tensor checks happen at submission).

The edit itself is deterministic and seed-free.

## Generation settings

Suite settings (from evaluation.json): temperature 1.0, top-k 20, top-p 0.95, max_tokens 8192, thinking on, presence_penalty 1.5.

Local runs only: same settings but capped at 128 new tokens, seed 1000, transformers on CPU in float32 — a smoke test budget, not the suite.

## Public-validation comparison

Verified results (leaderboard, not local estimates):

- Parent baseline: **17.40% private / 38.34% validation** (config `public-thinking8k-ladder500-v1`).
- This edit: **not yet submitted** — no verified score exists at the time of this PR. The submission ID/URL will be added here after submission.

Local estimates (advisory only, NOT comparable to suite scores):

- The pinned v6 suite cannot be run locally: `delta evaluate` requires a served vLLM endpoint (GPU) for pinned configurations; the legacy local path does not apply. No local validation score is claimed.
- CPU smoke test, 3 prompts, original vs edited: both models generate coherently with no degeneration. On "What is 17 x 24?" the edited model chose a cleaner decomposition (17x20 + 17x4) than the original's vertical multiplication; on a word problem and a multiple-choice question the traces were near-identical. All traces truncate at the 128-token cap, so no final answers were produced by either model. This shows the edited weights load and behave; it is not evidence for or against a score change.

## Limitations and negative findings

- All direct x1.05 evidence is on stronger GRPO parents, not on the original parent. On the current top parent, similar doses (x1.035, x1.042) scored BELOW that parent (49.0% vs 51.0%) — the effect is parent-dependent and non-monotonic; a second x1.05 step elsewhere gave no further gain.
- Nearest original-parent data points: x1.25 gave 19.80% private (+2.4) but validation fell 38.34% -> 36.44%; x1.667 gave 15.2% private, below baseline.
- Local comparison is limited by CPU: 128-token cap, ~13 minutes of generation, no suite score. The authoritative result is the submitted evaluation only.

## Submission

Status: patch built and validated locally; submission pending (quota check and final approval). This section will be updated with the submission ID and verified scores when they exist.
