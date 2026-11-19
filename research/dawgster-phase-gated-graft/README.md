# Phase-gated "ready" circuit graft on ac147fa6

Follow-up to `research/dawgster-ready-graft` (PR #65), which scored 44.0 private versus the parent's 50.4.
This patch keeps that graft's two repurposed neurons and adds a gate so the push applies only while the model is still thinking.
Like the original, it is a closed-form weight edit (ridge probes plus algebra). It involves no training and no merging.

## Diagnosis of the failed graft

We ran the pinned public suite in the native prompt formats (thinking on, private sampler).
On that run, the ungated graft kept firing **after** the think block had closed:

- When the answer was complete, the "ready" state persisted.
- The graft then pushed `</think>` instead of `<|im_end|>`.
- 143 of 380 generative items ended in hundreds of repeated `</think>` tokens, ran to the 8k cap, and left an empty final answer.
- On the dev harness, BBH's length-cap rate went from 0% to 11%.

The parent model also re-emits `</think>` in about 23% of items. That is useful self-correction. Cutting answers at the second `</think>` costs 1.2 points on the public suite and 11 items on the boxed proxy, so this patch leaves it alone.

## Edit (layer 23 MLP, 2 neurons, 3 tensors)

- **Ready score `s`:** the same ridge probe as before, read at the layer-23 MLP input.
- **Phase probe `p`:** a new ridge probe at the same point.
  - Fitted on think and answer tokens from traces in 7 public formats plus the boxed proxy.
  - Held-out means: about 0.85 while thinking and about 0.13 while answering, in every format.
- **SwiGLU used as an AND gate.** Both probe biases are folded into the constant channel (dim 0).
  - Neuron 1: `gate = a(p − 0.4)`, `up = s`.
  - Neuron 2: `gate = a(p − 0.6)`, `up = s`.
  - `down` columns: `±κ·v/(a·0.2)`, where `v` is the minimum-norm targeted `</think>` push.
  - Net output: `[silu(a(p−.4)) − silu(a(p−.6))]·s·κv/(0.2a)`, which is approximately `κ·s·v·1[thinking]`.
- **Settings:** κ=4, a=25.
- **Build:** `scripts/phase_gate.py fit` and then `step`.

## Results (local; see aggregate_results.json)

| Check | Parent ac147fa6 | This patch |
|---|---|---|
| Official `delta public-eval`, thinking config (1 run) | 43.70 (platform: 43.45) | 43.59 |
| Fast harness: 380 generative public items, native formats, 4 seeds | 43.73 | 45.47 (+1.74, 95% CI [−0.75, +4.32]) |
| Boxed GSM8K/MATH/ARC proxy (510 items) | 61.6 (5 seeds) | 61.3 (3 seeds; −0.34, CI [−2.2, +1.6]) |
| Items with more than 5 `</think>` tokens (harness) | 1 | 1–4 (ungated graft: 31) |

The harness length-cap rate drops on IFEval (12.1% → 6.2%) and MBPP (11.7% → 6.2%). GSM8K is 2–3 points lower in both harnesses.
κ=2 gives −0.06 and κ=6 gives +0.86, with inconsistent per-group effects.

**Expectation:** the catastrophic failure is fixed and nothing regresses, but a gain is not established. The likely private effect is between 0 and +1.5.

## Tooling

- `scripts/pubeval.sh`: serves a checkpoint with vLLM and runs the official `delta public-eval` with the thinking evaluation config.
- `scripts/pubfast_gen.py` and `pubfast_score.py`: run in about 5 minutes per seed.
  - Prompts are taken from an official run.
  - Generation is one offline vLLM batch.
  - Rescoring feeds the cached outputs back through the official lm-eval filters and the WASI code sandbox.

This PR contains no weights, credentials, signed URLs, private data or raw generations.

## Verified result

The submission was `1020e203-502f-4101-bb7a-860114638f05` (version `fdad0374`).

| | Private test | Validation |
|---|---|---|
| This patch | **50.8** (rank 1) | 44.55 |
| Parent ac147fa6 | 50.4 | 43.45 |
| Ungated graft (#65) | 44.0 | — |

Gating recovered the 6.8 points the ungated graft lost. The +0.4 margin over the parent is within the ±2.2 run noise.
