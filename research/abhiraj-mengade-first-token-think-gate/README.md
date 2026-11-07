# Closing thinking at the first token with a max-margin `</think>` row edit

**Status: long shot.** This edit tied the current leader in local testing. It was not shown to beat it, and this is one submission to find out how it behaves on the private test. All scores below are local estimates on public data unless marked as leaderboard results.

## Hypothesis

Under server decoding, the score is roughly the sum over `t` of `P(close at t) * A(t)`. Most leaderboard entries push the model to close its thinking earlier. Closing partway through a trace can be destructive: the `</think>` row gain ×1.5 entry finished more items but scored lower than the original.

This experiment tests the other end, `t = 0`: close immediately, so all reasoning happens in the visible answer.

**This edit effectively skips the thinking phase.** It does not improve thinking. On every local item it closed at the first generated token.

Embeddings are tied, so `logit(t) = h · E_t`, and changing row `E_c` of `</think>` acts as a linear classifier on the final hidden state `h`. The edit is the minimum-norm `d` that satisfies:

- at think-open positions: `h · (E_c + d) >= max_other(h) + k_pos / g`
- at answer positions: `h · (E_c + d) <= max_other(h) - k_neg / g`

It is solved as squared hinge plus ridge with L-BFGS, using `g = 3`, `k_pos = 12` and `k_neg = 8`. Two other edits go with it:

- final norm gain ×3
- the seven Wait/Hmm embedding rows ×0.5, following earlier entries (3655, 11158, 13428, 13784, 77264, 84485, 85152)

## Attribution and access

- Submitted by `abhiraj-mengade`. The research and code were AI-assisted.
- Only public material was used: the public leaderboard, this repository, published checkpoints, and public benchmark datasets.
- There was no access to private test data. The private prompt format is unknown to this work.

## Lineage and settings

- Parent: original model `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`, content hash `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Active evaluation at the time of writing: `public-ladder500-8k-128-v1`.
- Local decoding: thinking on, T 1.0, top_p 0.95, top_k 20, presence penalty 1.5 on generated tokens, 8,192-token cap, base seed 1000.
- Compute: MLX (mlx-lm 0.32.2) on a 16 GB Apple Silicon laptop, batch 8. MLX sometimes aborted on Metal allocation limits. Runs resumed by item ID, so resumed items were sampled from a re-seeded generator.

## Reproduction

Put these files in an `analysis/` directory inside a workspace from `delta competition init`, and run from the workspace root.

```bash
python analysis/build_items.py
python analysis/collect_gate.py 500 90
python analysis/fit_gate.py 3 12 8 analysis/delta_g3.npy
python analysis/build_final.py experiment/model/model.safetensors experiment/final/model.safetensors analysis/delta_g3.npy
delta patch build --parent model --modified final/model.safetensors --output submission
python analysis/e2_gate.py final_fresh 1 none --parent experiment/final --cap 8192 --skip 20 --batch 8
```

Run the `delta patch build` line from inside `experiment/`, and give `experiment/final/` copies of the parent's config and tokenizer files. The fitted vector is included as `delta_g3.npy`, because refitting needs `gate_data.npz`, which is not committed. The item bank has 300 items, 60 each from GSM8K, MATH-500, ARC-Challenge, MMLU and CommonsenseQA, excluding items in the public validation selections.

Expected hashes (sha256):

| Artifact | Hash |
|---|---|
| `delta_g3.npy` | `6c5cd9b46bd90aaba0d6e56cf79ba8026477efc4286f385026932aaf89b7addb` |
| Edited `model.safetensors` | `e111a867abfe8aed784c228a39f913159d80f78c6593bccb01c99b471a9e9197` |
| `submission/manifest.json` | `2f3f2b70c6e084445ebfb63ac77b99806eba8619441b475eb61e811478e8902c` |
| `submission/weights.safetensors` | `a9d86016946d8e6e448ab7b2933396ee8e0218b839ebf45d0bd17407b98b067f` |

The patch changes two tensors and eight embedding rows. After BF16 rounding, the row-delta error is 0.00103 against `||d|| = 0.1476`, and the applied gain averages 3.0003.

## Evidence (local and advisory)

The public `delta` validation was **not** run on this candidate, so there is no local parent-versus-candidate comparison on the validation selections. The numbers below come from the separate item bank above, and they run far higher than leaderboard scores, so they are not comparable with them.

### Main comparison: 200 items not previously used at this cap, candidate loaded from its file

| | Leader `cd0b95bb` | This candidate | Gate refit on the leader |
|---|---|---|---|
| Score | 57.0 | 56.5 | 52.0 |
| Paired difference from leader (95% bootstrap CI) | | −0.5 [−8.0, +7.0] | −5.0 [−13.0, +3.5] |
| Truncated | 5.5% | 15.5% | 2.5% |
| Accuracy on finished answers | 60.3% | 66.3% | 53.3% |
| Mean tokens | 1,524 | 1,646 | 853 |

Correct answers out of 40 per benchmark:

| Benchmark | Leader | This candidate | Gate refit on the leader |
|---|---|---|---|
| GSM8K | 26 | 23 | 23 |
| MATH-500 | 19 | 18 | 20 |
| ARC-Challenge | 29 | 25 | 23 |
| MMLU | 21 | 24 | 19 |
| CommonsenseQA | 19 | 23 | 19 |

The candidate won 29 items and lost 30. Stacking the gate on the leader made the score worse: truncation fell, but accuracy on finished answers fell further.

### Earlier comparison: 100 items, edit applied in memory

| | Original | Leader then, `b8af1dbc` | This candidate |
|---|---|---|---|
| Score | 39 | 52 | 61 |
| Truncated | 69% | 13% | 17% |

The difference from `b8af1dbc` was +9 [0, +18]. This set informed the choice of candidate, so it is not an independent test. The harness reproduced the leaderboard's ordering of the original and `b8af1dbc`, with a truncation rate for the original (69%) close to the leaderboard's (about 75%).

### Prompts without an answer-format instruction: same 100 items

| | `b8af1dbc` | This candidate |
|---|---|---|
| Real scoring rules | 20 | 15 |
| Lenient re-score, counting any clear answer | 58 | 58 |

The paired difference under the real rules is −5 [−12, +1]. Most multiple-choice answers name a letter in a form the scorer does not read, such as `**Correct Answer: B**`. The lenient re-score is an ad hoc check of my own, not an official scorer.

## Negative findings and limitations

- **No demonstrated gain over the leader.** The expected private result is near or below the leader, not above it.
- **Truncation may grow on the private test.** Both `b8af1dbc` and the current leader truncate much more there than locally. This candidate already truncates more than the leader locally, so it could lose ground.
- **Rumination moves rather than disappears.** Loops continue in the visible answer, and most truncated answers are repeated-line loops with no `\boxed`.
- **Forcing an exit from loops has little value.** An oracle cut each of 59 truncated answers at the first repeat and injected a closing phrase. The best phrase raised correct answers from 3 to 10, at most +2.3 points on 300 items, so no stop gate was built.
- **Format risk is unresolved.** On a private prompt without a format instruction, this candidate could score poorly.
- **Repeated use of the 200 items.** They were used for two candidate comparisons here, which is mild selection. The result was a tie either way.
- **Single scores are noisy.** After the move to `public-ladder500-8k-128-v1`, identical weights changed by up to 3.8 points.
- **Local harness differences.** This harness is MLX, not the server's vLLM, and its item mix and prompts differ from the private test.

## Submission

Pending. The ID and verified result will be added after submitting.
