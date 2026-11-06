# Repetition edit at 0.75 strength on cd0b95bb

## Hypothesis
A stronger published repetition edit (0.75 instead of 0.5) may reduce truncation on the current leader without losing accuracy.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material.

## Edit
- Parent: `cd0b95bb-c5c0-4656-b9be-ad6482f04075`. Evaluation configuration: `public-ladder500-8k-128-v1`.
- Tensors: `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight`.
- The parent carries the published edit from Sparks of Weight-Space Programming at 0.5 strength. `b8af1dbc` is the same stack without it for these tensors. The strength is rescaled linearly from their difference in float32 and cast to BF16.
- Lineage credit: poppingbean, SpacexCeo, xf3s, 0xZxh and the authors of the published edit.

## Reproduction
```sh
delta competition init qwen35-08b-v1 --parent cd0b95bb-c5c0-4656-b9be-ad6482f04075 --output leader
delta competition init qwen35-08b-v1 --parent b8af1dbc-d52b-4bbc-a302-3505d897303c --output no_rep
python build_arm.py r075 leader/model no_rep/model rep:0.75:0.75
delta patch build --parent leader/model --modified r075/model.safetensors --output submission
delta validate submission
```
Expected sha256: checkpoint `072093bc974f33bb9c86e7ff27c2321c10934708951bad04b41b4f0b316c7a3c`, patch `weights.safetensors` `b2c7ef6ef288b490ec07db6308729ef8e9f99e5fbd6ac924dc19200bbee17e9f`. Format: sparse-diff, 2 tensors.

## Evidence (local, advisory)
Proxy: `build_proxy.py` with seed 20261106 builds 320 prompts from public test splits (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-Challenge 40), used for measurement only. `run_proxy.py` runs vLLM 0.30.0 on an RTX 4080 with the configuration's sampling (thinking on, 8192 tokens, T 1.0, top-k 20, top-p 0.95, presence penalty 1.5), using the PyTorch sampler. Seeds 1000, 2000 and 3000, 3 samples each, 9 samples per prompt.

| Metric, candidate minus parent | Result |
|---|---:|
| Strict accuracy (correct, closed, at least 50 thinking tokens, not truncated) | +0.35 ± 0.81 |
| Truncation, points | −1.04 ± 0.59 |
| Accuracy on prompts both finished in all samples | −0.18 ± 0.99 |

Errors are standard errors across prompts. By seed against the parent's other seeds, strict is +2.29, −1.04 and −0.21. Public validation was not run locally.

## Limitations
The candidate is not distinguishable from its parent on this proxy. This entry tests the hypothesis and claims no improvement. Any rank gain may be luck, and the first seed alone overstated the effect.

## Negative findings from the same session
- `</think>` row gain 1.0, 1.1 and 1.2 instead of 1.3 on the parent (3 samples): strict −4.06, −3.33 and −1.25 against that seed.
- Another final-norm step ×1.05 on the parent (3 samples): strict −0.63 ± 1.26 against the parent's mean, shared accuracy −4.81 ± 1.76.
- The repetition edit in layer 15 only, at 0.5 and 1.0 strength (3 samples): strict −1.15 ± 1.19 and −0.73 ± 1.35 against the parent's mean.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Submitted, evaluation pending: https://deltaleague.ai/submissions/1b419ede-f60a-4210-bb8b-acc67db3fed1/
