# `</think>` row gain x1.5: a conditional closing bias through the tied embedding

Author: vdimarco. Agent-assisted (Claude), human-reviewed. One pre-registered arm, one
documented negative local finding (the additive version), no server runs yet.

## Problem

On `public-ladder500-8k-v2` the parent reaches the 8,192-token cap on 74.6% of generations.
Every board entry so far sits within noise of the parent on the private test, and truncation
moved with each edit. Locally the same thing shows up at small scale: on "What is 17
multiplied by 6?" and a true/false question the parent had not emitted `</think>` after 700
tokens. The dominant failure is no answer, not a wrong one.

## Hypothesis

Raising the `</think>` logit only at moments when the model already leans toward closing
shortens thinking, lowers truncation, and converts a share of capped generations into scored
answers. If the model closes before it has an answer, gsm8k, math500 and bbh fall while
truncation still drops. Both outcomes are informative.

## Mechanism

Qwen3.5-0.8B ties input and output embeddings (`tie_word_embeddings: true`) and has no logit
bias. `logit(t) = g(h) . E[t]` with `g(h)` the final RMSNorm output. Multiplying row
`E[</think>]` by a gain `s` multiplies that one logit by `s` and changes no other logit.

Why multiplicative rather than additive: the raw `</think>` logit is near zero while the model
ruminates and rises to +15..+26 only when it is about to close. A gain therefore adds
`(s-1) * logit` where it matters and nothing elsewhere.

Parent measurements (CPU, fp32, server sampling settings):

| quantity | value | source |
| --- | --- | --- |
| raw `</think>` logit during thinking, p10 / p50 / p90 / p99 | -4.1 / -0.4 / 5.3 / 15.7 | 1,264 positions, gsm8k trace |
| margin to the top logit, p50 / max | -24.3 / -4.9 | same |
| raw `</think>` logit at the 10 best-margin positions | 15 to 19 (top logit 20 to 27) | same |
| raw `</think>` logit at a natural close | 25.8 (runner-up 18.6) | prime-number MCQ, 323 tokens |
| positions in the stuck trace within 2 nats of the top after gain 1.2 / 1.3 / 1.4 / 1.5 | 1 / 2 / 2 / 3 | teacher-forced |

Chosen arm: `s = 1.5`. 1,024 bf16 entries change (every entry of the row), max relative
rounding error 0.38%. Row norm 0.571 -> 0.856.

## What the edit changes, teacher-forced parent vs edited

`verify_edit.py` (384 thinking positions, 4 prompts) and `natural_kl.py`:

| check | gain 1.5 |
| --- | --- |
| `</think>` logit shift | min -3.8, median +0.13, max +11.3 |
| max abs change on any other token's logit | 0.0 |
| KL(parent, edited) at the 3 positions after a natural `</think>` | 0.000, 0.048, 0.000 |
| natural prime-number run, seed 1000 | parent and edited token-identical, answer C |

Input side: the first layer applies RMSNorm to the embedding, which removes a pure scale.
The residual still carries the scaled raw embedding at that one position; the natural-context
KL above (at most 0.05 nats) is the measured cost. Forcing `</think>` mid-thought after 96
tokens gives KL up to 5 nats, but that context does not occur in deployment and is reported
only in `verify_gain15.json` for completeness.

## Negative local finding: additive bias along the mean hidden direction

First attempt: `E[</think>] += delta * mu_hat`, with `mu_hat` the unit mean of `g(h)` during
thinking (`measure_think.py`, `edit_think_bias.py`, `verify_bias3.json`). Target +3 nats
median shift; measured +0.45..+4.7, positive at every position, no other logit changed.
It is the wrong lever: the margin in a stuck trace never comes within 6 nats of the top, so a
+3 shift fires nowhere. On the 1,200-token gsm8k trace the additive edit produced a
token-identical truncated output. It also costs more on the input side (natural-context KL
not measured; forced-context KL 0.2..0.8). Not submitted. Files kept for the record.

## Pre-registered predictions (private test, `public-ladder500-8k-v2`)

Parent: validation 41.83, private 18.6 +/- 1.76, truncation 0.746, output tokens 3.52M.

1. Truncation rate falls below 0.65.
2. Output tokens fall by at least 10%.
3. Private test score is at or above the parent's 18.6.

Falsifier: private test below 15.1 (two marginal standard errors under the parent) means the
gain closes thinking before the answer is ready; the next arm, if any, is `s = 1.25`.
If 1 and 2 hold and 3 fails by less than that, same next arm. If 1 fails, the mechanism does
not reach the sampler as modeled and no further gain arm is planned.

## Lineage and settings

- Model `qwen35-08b-v1`, parent version `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`,
  parent content hash `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Evaluation config `public-ladder500-8k-v2`: temperature 1.0, top_p 0.95, top_k 20,
  presence_penalty 1.5, max_tokens 8192, thinking on.
- Local seeds: 1000 (+ prompt index in `measure_think.py`). Local sampling matched the server.
- Patch: `sparse-diff`, tensor `model.language_model.embed_tokens.weight`, indices
  254022656..254023679 (row 248069). `weights.safetensors` sha256
  `c823f08d287edead2a3214fc99224cde0c5e7f3ef3c1a2aed39fd18417a9a308`, 10,392 bytes.
- Edited checkpoint sha256 `50eab381049d501e023639f45c832dc532a3c4e6d8f799a056125c0df6bba3ae`.

## Reproduce

```
pip install 'deltaleague[local]'
delta competition init qwen35-08b-v1 --output experiment && cd experiment
cp -r /path/to/deltaleague/research/vdimarco-think-gain research
python research/edit_think_gain.py model 1.5 edited.safetensors
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
# optional checks (CPU, several minutes each)
python research/margin_profile.py model research/measure_parent.json research/margin_gsm8k.json 1200 "<gsm8k prompt in prompts.json[0]>"
python research/verify_edit.py model edited.safetensors research/verify_gain15.json research/prompts.json
python research/natural_kl.py model edited.safetensors research/natural_kl_gain15.json research/prompts_one.json 400
```

The edit itself is deterministic: scale one row by 1.5 and round to bf16. The measurement
scripts sample, so their numbers vary a little by machine; the JSON files used here are
committed.

## Local runs

CPU only (2 cores, no GPU), so no local public validation. Local generations check
mechanism, not score. The server result is the only evidence that counts.

## Access during research

Public parent checkpoint, public `validation.json` selection spec, CLI source, research
PRs #7 and #12, the leaderboard API. No private test data requested or seen. No benchmark
items were used to fit the edit; prompts in `prompts*.json` are hand-written.

## Submission

- Submission ID `aedb45f3-fee9-4827-8ef1-95e6fbe5f088`, submitted 2026-11-05 11:45 UTC with PR #16 attached.
- State at last update: verifying. Verified results will be added here.

## Limitations

- The gain also makes negative `</think>` logits more negative (min shift -3.8). Harmless
  for closing, but it is a change.
- Firing frequency was estimated on one stuck 1,264-token trace and one natural close.
- Presence penalty (1.5 on already-emitted tokens) favors `</think>` relative to repeated
  competitors; the firing estimate above ignores it, so it is conservative.
- One arm. If it helps, the dose-response is unknown.
