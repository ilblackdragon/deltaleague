# C1+ — half-strength repetition edit and final norm gain

## Hypothesis and lineage

Test the recipe of poppingbean’s leading version `cd0b95bb-c5c0-4656-b9be-ad6482f04075` using the gentler doubt-direction dose. The user specified the hypothesis and Codex assembled the checkpoint using public inputs.

The direct parent is xf3s’s gentler doubt-direction dose (×0.85), documented in [PR #22](https://github.com/ilblackdragon/deltaleague/pull/22). SpacexCeo’s [PR #24](https://github.com/ilblackdragon/deltaleague/pull/24) supplies the interpolation method and the reference version `344c8376-66a1-413b-abca-da2b5b986044`. SpacexCeo also contributed the ×1.05 final norm version `27e49fd4-d761-4d0b-aef3-abf1fadacd0b`. The final norm is copied bit for bit from SpacexCeo’s ×1.05 version, as in poppingbean’s [PR #43](https://github.com/ilblackdragon/deltaleague/pull/43). The underlying sparse weight edit comes from the authors of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

- Model `qwen35-08b-v1`
- Parent `d6b5f7e3-6757-46e4-a5ae-14e40b8252ed`
- Evaluation `public-ladder500-8k-128-v1`

## Forecast before submission

Treat an absolute private-test difference of less than 3 points from the parent’s 34.8 as noise. The short local inference check only tests whether the candidate can generate a coherent response.

## Reproduction

Run from this research directory. Use Python 3.12 and `deltaleague[local]`.

```sh
research_dir="$PWD"
delta competition init qwen35-08b-v1 --parent d6b5f7e3-6757-46e4-a5ae-14e40b8252ed --output exp-c1
cd exp-c1
delta model pull a9340e14-190b-473a-952f-65722b37d576 --output reference-norm
delta model pull 27e49fd4-d761-4d0b-aef3-abf1fadacd0b --output donor-norm
mkdir -p upstream
url=https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit
curl -fsSL "$url/manifest.json" -o upstream/manifest.json
curl -fsSL "$url/edit.npz" -o upstream/edit.npz
python "$research_dir/rebuild.py" model/model.safetensors upstream edited.safetensors 0.5 reference-norm/model.safetensors donor-norm/model.safetensors
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
delta run 'In one sentence, explain why a bicycle slows down when you stop pedaling.' --workspace . --weights edited.safetensors --device mps --no-thinking --seed 1000 --max-new-tokens 96 --json
```

The script checks the BF16 hashes of both parent tensors and reconstructed upstream tensors. It computes `W + 0.5 * (W_published - W)` in float32 and casts back to BF16. The interpolation uses no randomness or fitting.

## Local evidence

The parent’s final norm is bit-identical to a9340e14, SHA-256 `e1dfccb2ef4edd8c809d960b13c585d727cdc87d8dd78137e0f2bc2daa19c251` over tensor bytes. The transplanted norm matches 27e49fd4 bit for bit. The two published-edit tensors passed the upstream checks. The resulting three-tensor patch passed `delta validate`. The candidate generated a coherent response in 38 tokens with thinking disabled.

- Full checkpoint SHA-256 `65012a06da55ea393aa26a0efda44810984edb47e8a56b6b57bdb90e3135dbb0`
- Patch payload SHA-256 `fc47e21d60064ceff3ac79f022d1c459d77d9bdc37850a6e3eb9ce19c4f0c529`

## Official submission

Submission `b4a923b1-9ea2-4fc9-9cbb-5601278fb379` was verified as `ad8217e2-24ac-490f-afa6-d1bdf6ff1ab2` under `public-ladder500-8k-128-v1`.

| Metric | Parent | Candidate |
| --- | ---: | ---: |
| Private test | 34.8 | 33.6 |
| Public validation | 41.82 | 41.79 |
| Truncation | 28.4% | 18.4% |

Private-test difference: -1.2 points. This is within the prespecified 3-point noise threshold. The candidate ranked 6 at this leaderboard check.
