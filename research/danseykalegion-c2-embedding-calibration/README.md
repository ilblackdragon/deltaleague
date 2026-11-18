# C2: Minimal embedding-only completion calibration

## Research question

Can a very small embedding edit reduce common reasoning-loop markers without breaking basic output completion?

## Scope and prior work

This experiment follows public Delta League research on Wait/Hmm damping and think-closing behavior. It does not claim a new editing method. Its contribution is a deliberately small combined coefficient setting and an honest record of the current evaluation limitation.

## Lineage

- Model: `qwen35-08b-v1`
- Parent version: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`
- Parent content hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`
- Evaluation version: `public-ladder500-8k-v2`
- CLI: `deltaleague==0.1.1`
- Seed: `1000`

## Edit

Only `model.language_model.embed_tokens.weight` changes.

| Token | Token ID | Operation |
| --- | ---: | --- |
| `Wait` | 13784 | multiply row by 0.98 |
| ` wait` | 3655 | multiply row by 0.98 |
| `Hmm` | 77264 | multiply row by 0.98 |
| ` hmm` | 84485 | multiply row by 0.98 |
| `</think>` | 248069 | multiply row by 1.005 |

The output full-checkpoint SHA-256 was `32633344d8c58b0c1c4bce6cf1597aa9d1c1e947d9183150ecbc7e45128773fd`.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --output experiment
cd experiment
python /path/to/research/danseykalegion-c2-embedding-calibration/edit.py
delta patch build --parent model --modified candidate-c2.safetensors --output candidate-c2-submission
delta validate candidate-c2-submission
```

The validated sparse artifact contains one changed tensor. Model weights and patch files are intentionally excluded from this record.

## Local checks

On CPU, with `--no-thinking --max-new-tokens 32 --seed 1000`, the parent and C2 both returned `201` for the prompt `What is 17 times 23? Answer with only the number.`. The parent used 6 tokens in 11.32 seconds. C2 used 6 tokens in 15.56 seconds.

This one-prompt result is not a benchmark score and does not show improvement.

## Public validation limitation

The active server validation file is hash-verified but uses grouped `text`, `code`, and `bfcl` keys. CLI 0.1.1 expects a flat benchmark inventory and rejects it before generation with `Public validation benchmark inventory mismatch`. No public validation result is claimed.

## Limitations

- Submission `7cd5e3b6-3b45-4525-ab0d-b366dc1cd823` was verified as version `b7d176ac-27a3-46d8-80c5-dcd1ff042fe7` with research PR https://github.com/ilblackdragon/deltaleague/pull/25.
- Under `public-ladder500-8k-128-v1`, C2 scored 41.4109 public validation and 17.2 private test. The original model scored 41.3316 public validation and 17.4 private test. C2 ranked 17th at verification.
- The public gain (+0.0794) did not carry to private test (-0.2). This candidate is not an improvement and must not be resubmitted unchanged.
- CPU checks are not comparable to server evaluation.
