# little Qwen v0.1: take the doubt out of what the early layers write

Research by three (GitHub `xf3s`) with umber, an AI agent (Claude Opus 5.5 in Claude Code) that ran the experiments.
Done 2026-11-02 to 11-03, before Delta League existed. Submitted here 11-05.

## Hypothesis

When Qwen3.5-0.8B thinks, it mostly doesn't fail by being wrong. It fails by never stopping: it reaches an answer,
then reconsiders ("Wait, …", "But …", "Maybe …") until the token budget runs out. On Delta League's starter config the
original model truncates 74.6% of test generations at 8,192 tokens.

We think the reconsidering comes in two separable pieces:

1. **The word.** Layer 16 MLP neuron 998 writes nearly all of layer 16's push toward "Wait". It's ~0 in the prompt
   and fires at line starts in the thinking trace, gated on by thinking mode.
2. **The doubting itself.** A direction in the layer-13 residual stream separates line starts where the model is
   about to reconsider from line starts where it is about to compute. Silencing neuron 998 alone only moves the
   doubt to other words ("If", "Maybe", "Final check of the question text for trickery:"). Removing the direction
   is what stops it.

## The edit (29 tensors)

1. `model.language_model.layers.16.mlp.down_proj.weight[:, 998] *= -2`
2. For each of `self_attn.o_proj`, `linear_attn.out_proj` and `mlp.down_proj` in layers 0–13:
   `W <- W - d (d · W)`, where `d` is the unit doubt direction (`doubt_direction_L13.json`). Those layers can then
   no longer write anything along `d`.

`d` is a difference of means of layer-13 activations at thinking-trace line starts (reconsider minus compute). It's
the bisector of two fits, one on 8 problems and one on 30 (cosine 0.97 with each). Both fits used our own dev set:
30 problems from **MATH-500** (`HuggingFaceH4/MATH-500`, `Random(1102)` sample of 60, first 30 dev, last 30 held
and still unscored). The direction was fit from activations only, never from answers or scores.

**Overlap with Delta League's data, disclosed:** MATH-500 is also Delta League's `math500` benchmark. Three of our 30
dev problems are in the public validation rows (MATH-500 indices 128, 274, 455), and 455 is one of the 8 problems the
first fit used. One held problem is also in public validation. We can't see whether the private test includes any of
our dev problems. The edit is a single 1,024-number direction plus one neuron's scale, so we don't think it can carry
answers, but the math500 score is the one to read with this in mind. The dev indices are in `dev_indices.json`.

Rebuild from the Delta League parent, bit-exact with the submitted weights:

```sh
python build_edit.py model/model.safetensors doubt_direction_L13.json edited.safetensors
delta patch build --parent model --modified edited.safetensors --output submission
```

`build_edit.py` prints the sha256 of each changed tensor (BF16 bytes). Expected: see `tensor_hashes.txt`.

## What we measured before Delta League (our harness: vLLM, thinking on, k=4, seed 1102)

| set | cap | original | v0.1 |
|---|---|---|---|
| MATH dev (30 problems × 4) | 6,144 | 1/120 | 40/120 |
| MATH dev (30 × 4) | 16,384 | 17/120 | 54/120 |
| GSM8K (60 × 4, pre-registered, recipe fixed beforehand) | 16,384 | 43/240 | 120/240 |
| AIME 2024+2025 (60 × 4) | 16,384 | 0/240 | 0/240 |
| MATH dev, thinking **off** (harm check) | 6,144 | 42/120 | 45/120 |

Mean thinking length on GSM8K fell from ~15k to ~7.5k tokens, and "Wait" went from ~170 per run to almost none.

## Negative findings and limits

- **The dose is a sharp peak at exactly "remove."** Projection ×0.5: 22/120. ×1: 36–40. ×1.5: 1/120. ×2 (pushing
  toward "compute"): 0/120, with the model stuck repeating "But 3: / But 3." So more isn't better: flipping the direction breaks the model.
- **The edited model loops more when it does get stuck.** Without doubt it can't climb out of a repeated line: 24
  grooves, 15 escaped, never via "Wait". Unedited: 83 grooves, 82 escaped, 72 of them starting with "Wait". A
  calibrated "needle-lifter" neuron that fires on repetition halved the grooves, but the score didn't rise.
- **It fixes doubt, not capability.** AIME stays at 0.
- **The direction was fit on dev problems**, so the dev numbers may flatter it. The GSM8K run is the cleaner test.
- **We couldn't reproduce Delta League's scorer.** We rebuilt the 14 text tasks with lm-eval 0.4.13 + vLLM at the
  published indices (doc hashes match). Our harness gives the parent 26.6 on GSM8K where the platform gives 59.4, so
  we report no local Delta League *scores*. The one local measure below is finish rate, which doesn't depend on the
  scorer. The platform's scores are the result.
- **Not a decoding knob.** Unlike the current top entries (embedding rows for Wait/Hmm ×0.6, final-norm gain), this
  edit changes no embedding, no norm and no output row. We also submit it stacked on the current leader
  (`a9340e14`, which touches only `embed_tokens` and the final norm, so there's no overlap) to see whether the two
  add up. On the board, test score tracks how often generations finish within 8,192 tokens (original 75%
  truncated, leader 39%), and finishing is what this edit targets.
- **Relation to the published repetition edit (layers 15/19):** its main direction has cosine 0.03 with `d`, so
  they're different things. The published edit lines up better with the "stuck" direction we found
  separately (0.37 / 0.46).

## Submissions and verified results (2026-11-05)

All three were submitted as exact rank-1 LoRA patches (452 KB): every changed tensor's delta is rank 1. `make_lora.py`
builds them from a `delta model pull` folder; with SCALE 1.0 its output is byte-identical to submission A's
upload and reconstructs v0.1 bit-exactly.

| submission | parent | version | private test | truncated | public val |
|---|---|---|---|---|---|
| A: v0.1 alone | `0ea62f40` (original, 18.6) | `85e98a7f-3056-4f32-89eb-988aa4a3981c` | 27.6 | 36.6% | 40.7 |
| B: v0.1 stacked on the leader | `a9340e14` (31.4) | `b8af1dbc-d52b-4bbc-a302-3505d897303c` | **36.0** | 23.0% | 41.6 |
| C: like B, projection ×0.85 | `a9340e14` (31.4) | `d6b5f7e3-6757-46e4-a5ae-14e40b8252ed` | 33.2 | 23.2% | 41.7 |

Their test standard error is about 2.1 per entry.

**What we predicted, before any score came back:** C > B > leader > A. B > leader > A held. **C > B did not.**

**Local check that chose C (an estimate, not a score):** we measured finish rate on the 128 public gsm8k + math500
rows (zero-shot, the platform's sampling, cap 8,192, k=2, 256 runs each). Finish rate doesn't depend on the scorer.

| model | finished | right (math-verify) |
|---|---|---|
| original | 13 | 13 |
| v0.1 | 112 | 91 |
| leader | 101 | 96 |
| leader + v0.1 (B) | 186 | 127 |
| leader + v0.1, projection ×0.85 (C) | 173 | 142 |
| leader + v0.1, projection ×0.75 | 174 | 138 |

This ordered A, the leader and B correctly, but its preference for ×0.85 didn't carry over: on the private test C
truncates as often as B (23%) and scores 2.8 lower, about one standard error. Our check covered only math, while the
test mixes 18 kinds of tasks. Our zero-shot prompt also makes the original finish far less often (5%) than on the
platform, so only the ordering is meaningful.

**Unexplained:** on the public validation rows, all three of our entries score low on math500 (28–30, against
the leader's 50) and a little low on gsm8k (42–50 vs 52), which doesn't match their test scores. With 64 rows and one
sample each this may be noise. We haven't looked further.

Evaluation config `public-ladder500-8k-v2`.
