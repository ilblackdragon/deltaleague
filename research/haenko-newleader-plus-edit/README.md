# Combine the Wait/Hmm leader with the published repetition edit

## Hypothesis

Do two edits aimed at reducing extended reasoning compose? Apply the published sparse edit by the authors of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) to the embedding and final normalization edit by roshaans. The user selected this combination and Codex implemented the local check using public data.

- Model: `qwen35-08b-v1`
- Parent: `ab592312-ec77-4f98-b17f-6ea17723b74e` (roshaans, “Dampen Wait/Hmm + hedge token embeddings (0.6) with final norm gain x1.4”)
- Evaluation: `public-ladder500-8k-v2`
- Published edit revision: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`

## Prediction recorded before evaluation

The candidate will reduce the official truncation rate below the leader's 52.6%. Treat an absolute private-test difference of less than 4 points from the leader's 28.2 as noise for this experiment. This is a preselected interpretation threshold.

## Reproduction

Run from this research directory with Python 3.12, `deltaleague[local]==0.1.1`, and `datasets`.

```sh
research_dir="$PWD"
delta competition init qwen35-08b-v1 --parent ab592312-ec77-4f98-b17f-6ea17723b74e --output experiment-newleader-plus-edit
cd experiment-newleader-plus-edit
mkdir -p upstream
upstream=https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit
curl -fsSL "$upstream/manifest.json" -o upstream/manifest.json
curl -fsSL "$upstream/edit.npz" -o upstream/edit.npz
python "$research_dir/rebuild.py"
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
python "$research_dir/smoke.py" . --prepare
python "$research_dir/smoke.py" .
python "$research_dir/smoke.py" . --candidate
```

The unmodified upstream rebuild script verifies the SHA-256 of both parent tensors before replacing their entries and verifies each result against the pinned manifest. The targeted tensors are the MLP down projections in layers 15 and 19. The rebuild changed 65,260 and 64,928 entries respectively. The verified patch contains two tensors.

## Local completion check

Use the first two manifest selections from each of GSM8K and MATH500, with document hashes checked against the public manifest. Both checkpoints use seed 1000 and the settings from evaluation.json, including thinking and the 8192-token cap. The local prompts ask for a boxed final answer. The selected dataset indices are GSM8K 3 and 25, and MATH500 2 and 7. For MATH500 index 7 the Minerva normalization removes the degree suffix from the reference answer before hashing. Reference answers are not used for scoring. Report whether reasoning closes and a final answer follows, with token count and cap hits. The parent and candidate both loaded on Apple MPS. Each row gives the number of generated tokens and whether a boxed final answer followed the closing thinking tag.

| Public problem | Parent | Candidate |
| --- | --- | --- |
| GSM8K 3 | 8192, capped | 940, finished |
| GSM8K 25 | 5323, finished | 4412, finished |
| MATH500 2 | 8192, capped | 4312, finished |
| MATH500 7 | 8192, capped | 8192, capped |

The parent closed thinking and produced a final answer on 1 of 4 prompts. The candidate did so on 3 of 4 prompts. Total generation time was 45.2 minutes. This is a local completion check. After submission, we checked the final answers against the reference answers for these same four public items.

Patch artifact SHA-256: `1fb40231464e2af5ee22ac2d101ea2ef70e9db4a1adf64db47aa88b1d47a61a1`.

| Public problem | Reference | Parent final answer | Candidate final answer |
| --- | --- | --- | --- |
| GSM8K 3 | 540 | Missing (capped) | 180 (incorrect) |
| GSM8K 25 | 2 | 2 (correct) | 2 (correct) |
| MATH500 2 | 14/3 | Missing (capped) | 14/3 (correct) |
| MATH500 7 | 90 degrees | Missing (capped) | Missing (capped) |

The candidate produced two correct answers and one incorrect answer. The parent produced one correct answer. This four-item check is too small to estimate benchmark accuracy.

## Official submission

Delta League submission `0235948c-59ca-46c1-abf9-3d64a096a9fb` was verified as version `9ae2205f-3b0b-49db-a01f-0dfef287e5d0`.

| Official metric | Parent | Candidate |
| --- | ---: | ---: |
| Truncation rate | 52.6% | 50.4% |
| Private test | 28.2 | 26.2 |
| Public validation | 41.02 | 41.13 |
| Leaderboard rank at verification | 1 | 2 |

Truncation fell by 2.2 percentage points, meeting the forecast. The private-test score fell by 2.0 points, within the experiment's preselected 4-point noise threshold.
