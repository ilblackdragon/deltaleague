# Soft deadline: a length-aware `</think>` bias through a rank-1 edit of layer 23

## Hypothesis
On the current leader, about 10% of private-style generations never close `</think>` within 8,192 tokens, and they are mostly re-verification loops, not verbatim repetition. Earlier edits raised the `</think>` row uniformly, which trades early closes for late ones. A bias that grows with the length of the thinking phase should close only the runs that have gone on too long, so truncation falls without pushing short runs to close early.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## The edit
- Parent: `cd0b95bb-c5c0-4656-b9be-ad6482f04075`. Evaluation configuration: `public-ladder500-8k-128-v1`.
- One tensor: `model.language_model.layers.23.mlp.down_proj.weight` gets `W += 3.0 * v_close * u_pos^T`.
- `u_pos` (3,584-d, unit) is a direction in the MLP intermediate of layer 23 whose projection grows with the position inside the thinking phase. It was fitted by least squares on teacher-forced hidden states of 24 parent generations on public prompts (`probe_states.py`, `fit_edits.py`); held-out correlation with position 0.52, mean projection −0.02 at 0–500 tokens rising to +0.11 past 5,000, within-bucket standard deviation 0.035.
- `v_close` (1,024-d, unit) is the tied embedding row of `</think>` scaled by the final-norm gain `(1 + w)`, so the added residual raises the `</think>` logit and little else: at scale 1 the mean log-probability of generated tokens changed by less than 0.001 (`calibrate_rank1.py`).
- Net effect: the `</think>` logit is lowered slightly early in thinking and raised by a few logits after several thousand tokens. Nothing was fitted to evaluation data; the direction comes from the parent's own generations on public prompts.
- Both vectors are in `directions_soft_deadline.json`. Lineage credit: poppingbean, SpacexCeo, xf3s, 0xZxh.

## Reproduction
```sh
delta competition init qwen35-08b-v1 --parent cd0b95bb-c5c0-4656-b9be-ad6482f04075 --output leader
python build_rank1.py leader/model directions_soft_deadline.json pc3 pos_close:3.0
delta patch build --parent leader/model --modified pc3/model.safetensors --output submission
delta validate submission
```
Expected sha256: checkpoint `d673e0cb1d91394a85a76d90618b522b730a420a4753187ec65f66a4dd30d4b8`, patch `weights.safetensors` `1d6307761a3847d0042e35fd1e384112ce5fce40f599aa1aac798e467c026248`. Format sparse-diff, one tensor. The directions file reproduces the fitted vectors to 7 decimals; refitting from scratch needs the parent's generations (`run_proxy.py`, then `probe_states.py` and `fit_edits.py`).

## Evidence (local, advisory)
Proxy: `build_proxy.py`, seed 20261106, 320 prompts from public test splits (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-Challenge 40), used for measurement only. `run_proxy.py` on an RTX 4080 with vLLM 0.30.0, the configuration's sampling (thinking on, 8,192 tokens, T 1.0, top-k 20, top-p 0.95, presence penalty 1.5), PyTorch sampler. Seeds 1000, 2000 and 3000, 3 samples each: 9 samples per prompt for both the parent and the candidate. Strict: correct, `</think>` closed, at least 50 thinking tokens, not truncated.

| Candidate minus parent, 9 samples per prompt | Result |
|---|---:|
| Strict accuracy | **+1.63 ± 0.93** |
| Truncation, points | **−4.76 ± 0.77** |
| Never-closed share | 9.8% → 3.1% (seed 1000) |
| Answer-phase overflow | 6.4% → 8.3% (seed 1000) |
| Accuracy on prompts both finished in all samples | −1.2 ± 1.1 |

Per seed, same-seed strict: 49.6 vs 48.3, 48.4 vs 45.8, 47.5 vs 46.5. By benchmark: MATH-500 +2.4 ± 1.4, MMLU-Pro +3.5 ± 1.8, ARC-Challenge +1.4 ± 3.3, GSM8K −1.3 ± 1.9. Errors are standard errors across prompts.

Dose: scale 6 closes runs mid-thought (overflow 31%, strict −8.7 ± 1.6 on seed 1000) and was rejected. Scales 2 and 4 are being measured.

## Limitations
The strict gain is about 1.8 standard errors and maps to roughly +0.8 on the private scale, so the rank outcome is mostly noise. The truncation drop is the robust part. Some rescued runs now overflow in the answer phase instead. Public validation was not run locally.

## Negative findings from the same session
- A rank-1 edit pushing the `\boxed` token in the answer phase of overflowing runs (direction AUC 0.81): no gain at scale 0.3 (−0.4 ± 1.3) and harmful at 0.6, because normal answers stop boxing. A logit bias does not create a conclusion the model lacks.
- Repetition-edit strengths 0.625, 0.75 (verified: private 33.6) and 0.875 on this parent: all within noise of the parent.
- Another final-norm step (×1.05), and the repetition edit in layer 15 only: no gain.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Pending.
