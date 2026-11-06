# C1 — half-strength published repetition edit

## Hypothesis and lineage

Test whether half of the published repetition edit adds value to the gentler doubt-direction removal. The user specified the hypothesis and Codex assembled the checkpoint using public inputs.

The direct parent is xf3s’s gentler doubt-direction dose (×0.85), documented in [PR #22](https://github.com/ilblackdragon/deltaleague/pull/22). SpacexCeo’s [PR #24](https://github.com/ilblackdragon/deltaleague/pull/24) supplies the interpolation method and the reference version `344c8376-66a1-413b-abca-da2b5b986044`. SpacexCeo also contributed the ×1.05 final norm version `27e49fd4-d761-4d0b-aef3-abf1fadacd0b`. The underlying sparse weight edit comes from the authors of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

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
mkdir -p upstream
url=https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit
curl -fsSL "$url/manifest.json" -o upstream/manifest.json
curl -fsSL "$url/edit.npz" -o upstream/edit.npz
python "$research_dir/rebuild.py" model/model.safetensors upstream edited.safetensors 0.5
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
delta run 'In one sentence, explain why a bicycle slows down when you stop pedaling.' --workspace . --weights edited.safetensors --device mps --no-thinking --seed 1000 --max-new-tokens 96 --json
```

The script checks the BF16 hashes of both parent tensors and reconstructed upstream tensors. It computes `W + 0.5 * (W_published - W)` in float32 and casts back to BF16. The interpolation uses no randomness or fitting.

## Local evidence

Both layers matched the upstream parent and reconstructed published hashes. At half strength, 64,904 entries changed in layer 15 and 64,448 in layer 19. The two-tensor sparse patch passed `delta validate`.

The candidate produced a coherent sentence in 38 tokens on Apple MPS with thinking disabled. This check did not score answer quality.

- Full checkpoint SHA-256 `e5c80918871203f37a1978233abf640f94e8607524d747e98b1796ff32776750`
- Patch payload SHA-256 `291996a885cc8c3b962023cfce0b5e3e6ac25d4ae335db207d0e5435c5475e60`

The patch payload is shared with PR #24 because the two edited tensors start from the same upstream weights. The full checkpoint uses a different parent and has a distinct hash.

## Official submission

Submission `cee99a34-59f1-43ad-bbd9-63d26466b73d` was verified as `3a69b3a9-8c37-4a72-a5af-59423299ff28` under `public-ladder500-8k-128-v1`.

| Metric | Parent | Candidate |
| --- | ---: | ---: |
| Private test | 34.8 | 32.0 |
| Public validation | 41.82 | 41.20 |
| Truncation | 28.4% | 19.4% |

Private-test difference: -2.8 points. This is within the prespecified 3-point noise threshold. The candidate ranked 10 at this leaderboard check.
