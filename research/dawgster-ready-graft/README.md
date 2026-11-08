# Ready-to-answer circuit graft on the leader (`ac147fa6`)

A gradient-free, hand-compiled circuit stacked on the current leader. Inspired by
[circuit grafts](https://lf.wtf/research/circuit-grafts/): a linear read of a hidden feature, compiled into a few repurposed
neurons, writing a targeted push onto one output token. No training; one least-squares fit plus one 1024×1024 linear solve.

Research was done by Claude Code (Claude Opus 5.5) working for @dawgster, using only public material: the public
leaderboard/API, this repository and its PRs, public GSM8K / MATH-500 / ARC-Challenge test splits, and the leader's weights.

## Hypothesis

On the leader, ~22% of MATH-style items end with `</think>` emitted mid-derivation (median ~3.8k thinking tokens vs
~1.6k for finished items), followed by an answer section that keeps deriving until the 8k cap (score 0). A switch that
reads "has the model reached an answer?" and writes onto the `</think>` logit only should lower premature closes and
raise well-timed ones, unlike global `</think>` row scaling, which trades truncation for accuracy.

## The edit (3 tensors in layer 23, 2 neurons)

1. **Read.** Ridge regression (λ=10, ±1 targets) on the layer-23 MLP input. Positives: last 150 thinking positions of
   single-close, correct traces. Negatives: 60 positions sampled from 20–70% of each trace. Held-out accuracy 0.809.
2. **Neurons.** The two least-used layer-23 MLP neurons (1998, 3529; mean |act|·‖down‖ ≈ 0.007) are overwritten:
   gate = ±s, up = 1/mean(x₀) · e₀ (dim 0 is a large near-constant channel, mean −13.3), so the pair outputs
   silu(s) − silu(−s) = s exactly. The probe bias is folded into the dim-0 gate weight.
3. **Write.** Down columns = ±κ · v with v = (EᵀE + λI)⁻¹ E[`</think>`] / (1+w_final) · rms, normalised to +1 logit on
   `</think>` per unit of s. Off-target leakage: max 0.35 logit/unit, rms 0.02. κ = 4.

Layer 23's MLP only feeds the final norm, so the edit cannot change any later token's input representation (unlike
row edits on the tied embedding).

```
python scripts/build_proxy.py                                   # 510 items: 170 GSM8K, 170 MATH-500, 170 ARC-C (seed 1108)
python scripts/evalproxy.py models/leader runs/leader --seeds 1,2,3,4,5
python scripts/graft_ready.py fit models/leader "runs/leader/seed[123].jsonl" runs/graft_ready.pt
python scripts/graft_ready.py build models/leader runs/graft_ready.pt models/gr_k4 4
python scripts/evalproxy.py models/gr_k4 runs/gr_k4 --seeds 1,2,3,4,5
python scripts/cmp.py runs/leader runs/gr_k4
delta patch build --parent models/leader --modified models/gr_k4/model.safetensors --output submission
```

Hashes: parent content `8141721bc6ebf6a0239c7c11b2a489d0665735357b2852addce81b73d437f415`; edited
`model.safetensors` sha256 `c62db3733b2086ecb6b9c768d464cdd6323aa9b1d0678d404888c0dc208b989f`; patch
`weights.safetensors` sha256 `246aa01d8896c3e781960a77657cb31c2d68e366314431de8b48c504c29d7f8b` (sparse-diff, 3 tensors).

Compute: one RTX 3090 (24 GB), vLLM 0.33, ~4.5 min per proxy seed.

## Evidence (local estimates only, not verified results)

Proxy = private-test sampler from `public-thinking8k-ladder500-v1` (thinking on, T=1.0, top_p 0.95, top_k 20,
presence 1.5, 8192 tokens), prompt + `\nPlease reason step by step, and put your final answer within \boxed{}.`,
scored with `delta_cli.scoring.score_answer`. Per-seed numbers in `aggregate_results.json`.

| | overall | GSM8K | MATH-500 | ARC-C | truncated |
|---|---|---|---|---|---|
| leader `ac147fa6` (5 seeds) | 61.6 | 70.4 | 39.5 | 75.1 | 10.7 |
| + ready graft κ=4 (5 seeds) | 62.4 | 71.9 | 40.6 | 74.8 | 9.9 |

Paired by item, mean over seeds: **+0.78, 95% bootstrap CI [−0.78, +2.43]**. Not significant. The proxy is easier than
the private test (leader: 61.6 proxy vs 50.4 private). Public 18-benchmark validation was not run locally.

## Negative findings (same proxy, single seed unless noted)

- κ=2: 62.4; κ=8: 59.2 (over-push).
- Switch re-fit on the exact decision (good close vs close-then-truncate), κ=4: +0.4 (2 seeds); κ=8: 28.0 — blocking
  closes makes the model loop inside thinking (61% truncated). The premature-close items are lost either way.
- `</think>` row ×0.77 / ×0.885 / ×1.1: 61.0 / 61.6 / 61.4 (×1.1 halves truncation, same score).
- Within-class temperature on tied embedding rows (digits ×1.3, mean-preserving): ≈57 (more loops).
- Contrastive same-item "success direction" (solved vs failed attempts of the same item, layers 12–16) as a constant
  residual write: mainly a brevity direction; truncation falls to 0.6% but accuracy drops (56.7–60.6).
- Static depth extrapolation (scale residual writes of layers ≥20 ×1.15, ≥16 ×1.10): 60.0 / 60.6.
- Full-attention `q_norm` ×1.15 / ×1.3: 61.0 / 59.6.
- Sampler-side upper bounds (decode-time, not weight edits): presence penalty 0 → 60.1 (2 seeds); min_p 0.1 → 62.25;
  T 0.9 → 61.5. Temperature-like levers are close to saturated on this checkpoint.
- Diagnostics on leader traces: explicit arithmetic errors appear in few failures; multiple-choice format failures
  ≈0.8%; a correct answer in thinking but a wrong final answer ≈0.4% of ARC items.

## Submission

Submission `ba02dd30-c06e-46ae-9bb7-a0678a026030` (submitted 2026-11-08, parent `ac147fa6`). Verified result (model version `4e108bb5`): **private test 44.0** (parent 50.4), truncation 21.4% (parent 14.4%),
public validation 42.24 (parent 43.45). **The graft is harmful on the official evaluation (−6.4)** despite +0.8 on the local proxy.

Likely cause: the probe and its bias were fit only on proxy prompts (GSM8K/MATH-500/ARC-C with one instruction suffix). On
other prompt formats the score s shifts negative, so the graft becomes a net push *against* `</think>` and truncation rises
— the same failure seen locally at κ=8 with the decision-boundary switch. A switch like this needs calibration across prompt
distributions (or a bias-free, zero-mean write) before it can be trusted off-proxy. Recorded as a negative result.
