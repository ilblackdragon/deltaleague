# Leader 344c8376 + the ×1.05 final-norm gain from 27e49fd4

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `344c8376-66a1-413b-abca-da2b5b986044` (rank 1, private test 35.0).

## Hypothesis

Two edits sit on the same parent, `b8af1dbc` (test 32.8), and touch disjoint tensors:

| Entry | Change vs b8af1dbc | Private test |
|---|---|---|
| `344c8376` ([PR #24](https://github.com/ilblackdragon/deltaleague/pull/24), SpacexCeo) | `layers.15/19.mlp.down_proj.weight` only | 35.0 |
| `27e49fd4` (SpacexCeo) | `model.language_model.norm.weight` only (`(1+w)` ×1.0499) | 34.2 |

If the two effects are roughly additive, stacking them should beat the leader. The patch therefore copies `27e49fd4`'s final-norm tensor into `344c8376` and changes nothing else.

Credit: both source edits are SpacexCeo's work. This entry only combines them.

## Reproduce

```
python make_edit.py LEADER_344c8376.safetensors PARENT_b8af1dbc.safetensors X105_27e49fd4.safetensors OUT.safetensors
```

`make_edit.py` asserts two things before writing:

- `27e49fd4` differs from `b8af1dbc` only in the final norm.
- The leader's final norm equals `b8af1dbc`'s.

The output therefore differs from the leader in exactly one tensor (1,024 values).

| Artifact | sha256 |
|---|---|
| edited weights | `c12d3505fc9d846ba8582c3812ef752a4ed930b52916a8cd7d7eab5b80574c04` |
| patch manifest | `ef296f83272b40fadcd723a55fd2e5c7a3b089f21f3b7394dbcf3b3d6b263e40` |
| parent content hash | `a432d0987dd4ed9d31fa58422820e5f228f433a93fff629ed2fb5b1b122e2544` |

`delta validate` passes.

## Local check

Setup:

- 40 public gsm8k items (the first 40 by index, doc hashes verified).
- Private-test generation settings: T 1.0, top_p 0.95, top_k 20, presence 1.5, thinking on, 8,192-token cap.
- Seeds are paired per item (1000+position).
- Run on GPU in BF16 (AMD Radeon 8060S, ROCm).
- Scored with `rescore_gsm8k.py`. The CLI's math scorer returns 0 on Windows because math_verify's multiprocessing timeout fails; this script runs the same extraction in-process.

Commands:

```
python gsm8k_sampled.py WORKSPACE LEADER_DIR STACKED_DIR REPORT_PREFIX GENERATIONS.jsonl --n 40 --max-tokens 8192 --deadline-minutes 150
python rescore_gsm8k.py <workspace> generations.jsonl out
python compare_items.py out.base.json out.edit.json
```

| | Leader 344c8376 | Stacked |
|---|---|---|
| gsm8k accuracy | 55.0 | 52.5 |
| mean tokens | 1,551 | 1,251 |
| truncated / think not closed | 0 / 0 | 0 / 0 |

Paired difference: −2.5 points, 95% bootstrap CI [−20, +15], 6 wins and 7 losses, sign-test p = 1.0. Strict and lenient scoring agree, because every trace closed.

Results are in `results/` (aggregate and per-item 0/1 only; no generations).

## Honest reading

- **The local test shows no gain.** 40 items cannot detect a 1–2 point effect. The only clear signal is about 19% shorter reasoning, which matches the higher-confidence logits.
- **Private-test noise is about the same size as the hoped-for gain.** SE is about 2.1 per entry. PR #24's own README reports 33.0 for this model on another config, against 35.0 on the current leaderboard.
- **Extra norm gain has not always helped.** On `ab592312` (28.6), a further ×1.25 gave `86968401` (27.6). On `b8af1dbc`, ×1.05 gave `27e49fd4` (+1.4). The lineage's final norm is already ×1.40 of the original, so ×1.05 more is small but its effect is untested on this branch.

## Official result

Pending submission.
