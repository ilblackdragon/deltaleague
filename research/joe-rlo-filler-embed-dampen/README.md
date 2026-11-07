# Delta League — qwen35-08b-v1 — Experiment 1: dampen "Wait/Hmm" filler tokens

Author: Joe (joe-rlo) · prepared with Grok Bot · 2026-11-05 (America/New_York)

## Lineage / IDs
- Competition model: `qwen35-08b-v1` (Qwen3.5 · 0.8B, `tie_word_embeddings: true`)
- Parent version: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` ("Original model"), content hash `708085f8…d537b8`
- Parent `model.safetensors` sha256: `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` (verified unchanged after all edits)
- Evaluation config: `public-ladder500-8k-v2` (temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, thinking on, max_tokens 8192)
- CLI: deltaleague 0.1.1 (PyPI) + source `ilblackdragon/deltaleague@d5fb616` for local tooling; transformers 5.22.1, torch 2.15.0+cpu

## Context (leaderboard snapshot, fetched 2026-11-05 ~00:42 EST)
| rank | entry | val | private test | trunc rate |
|---|---|---|---|---|
| 1 | Dampen Wait/Hmm token embeddings (scale 0.4) — roshaans | 41.36 | 22.4 | 0.598 |
| 2 | Published weight-space edit reproduction — ilblackdragon | 41.26 | 19.0 | 0.716 |
| 2 | Final norm gain ×1.25 — abbyshekit (PR #7) | 41.45 | 19.0 | 0.756 |
| 4 | Original model | 41.83 | 18.6 | 0.746 |

Key observation: at an 8192-token budget the parent truncates ~75% of generations. The #1 entry is also the one with the lowest truncation rate. Most of the gain probably comes from shorter, less looping thinking traces, not from new knowledge.

## Hypothesis
Qwen3.5 has tied input/output embeddings, so scaling a token's embedding row by s also scales that token's output logit by s. Scaling the "Wait"/"Hmm" filler rows by 0.4 should make self-interrupting backtracking much rarer. That should give fewer runaway thinking traces, less truncation at 8K, and more scored final answers. It should not touch any other token.

- **Candidate A (primary):** an independent reproduction of the #1 idea. roshaans published no PR, so their exact token set is unknown.
- **Candidate B (novel combo):** A plus final-RMSNorm gain ×1.25 (#2-tied idea, PR #7). Both edits push in the same direction: B sharpens the distribution and A removes the filler tokens.

## Exact edit (scripts/make_edits.py)
- Tensor `model.language_model.embed_tokens.weight` [248320, 1024] BF16. Rows ×0.4 (computed in fp32, then cast back to bf16):
  `13784 "Wait"`, `13428 " Wait"`, `77264 "Hmm"`, `85152 " Hmm"`, `11158 "wait"`, `3655 " wait"`, `84485 " hmm"` → 7 rows = 7,168 entries changed.
- B only: `model.language_model.norm.weight` [1024]. Qwen3.5 RMSNorm computes `x·(1+w)`, so `w' = 1.25·(1+w) − 1` (1,024 entries).
- Deterministic, with no data fitting. The script asserts that the parent hash is unchanged.

```sh
cd /workspace/deltaleague-experiment
.venv/bin/python scripts/make_edits.py qwen35        # writes qwen35/edited.safetensors, qwen35/edited-B-norm125.safetensors
cd qwen35
delta patch build --parent model --modified edited.safetensors --output submission
delta patch build --parent model --modified edited-B-norm125.safetensors --output submission-B
delta validate submission && delta validate submission-B
```

| artifact | sha256 |
|---|---|
| edited.safetensors (A) | `02edd644cc9e220ccf84baaaa8f24a4273351c0be873af97ec40d9710eeeac04` |
| submission/weights.safetensors (A, sparse-diff, 71,832 B) | `d450165bb76c2e5ba2146dbef3c8cecc21a89eefa0259881cc0299de3fedb75a` |
| edited-B-norm125.safetensors (B) | `c4f2b535879e49b821478ddff38509c17371e97613f5d2dc943d38b943da16e4` |
| submission-B/weights.safetensors (B) | `34a2f5fabff467aac24d69f950d8b79a1f6d804f45eadf149ae5d573bd7c2486` |

Both patches pass `delta validate`. I also reconstructed both with `delta_cli.artifacts.reconstruct` and checked them against the full edited checkpoints: bit-exact.

## Local advisory evaluation
**Blocker:** in CLI 0.1.1 (and source main), `delta evaluate` fails with `Public validation benchmark inventory mismatch` on this config. The new `validation.json` lists selections by doc hash under `text/code/bfcl` instead of carrying prompts, so the official local evaluator could not run.

**Workaround:** `scripts/mini_eval.py` is a small batched harness. It reuses the CLI's `Runner` loader, the official `score_answer()` scorer, and the config's sampling settings (temp/top_p/top_k/presence penalty, thinking on). It overrides `max_new_tokens` only. Items come from `mini_items.json`, built from public datasets at the config's pinned revisions: gsm8k test 0–15, MATH-500 L≤3 ×8, ARC-Challenge ×8. CPU only: 8 cores, about 54 tok/s aggregate at batch 16. Seeds are per batch, not per example.

| arm | items | max_new_tokens | acc | trunc | closed `</think>` | Wait/Hmm tokens per 1k | other hesitation words per 1k |
|---|---|---|---|---|---|---|---|
| baseline | gsm8k ×16 | 2048 | 6.2% (1/16) | 100% | 0% | 4.52 | 5.34 |
| baseline | gsm8k ×16 | 512 | 0% | 100% | 0% | 3.66 (30 total) | 4.39 |
| **A** | gsm8k ×16 | 512 | 0% | 100% | 0% | **0.00 (0 total)** | 0.61 |
| **B** | gsm8k ×16 | 512 | 0% | 100% | 0% | **0.00 (0 total)** | 0.61 |

Reading the results:
- **What the run shows:** the edit works mechanically. Filler tokens went from 30 occurrences to 0 in matched runs. Outputs stayed coherent on inspection, and other hedges ("But", "Actually", "Alternatively", "No,") kept roughly baseline frequency in A. B produced somewhat fewer of them.
- **What it can't show:** accuracy. On CPU, every arm truncates 100% even at 2048 tokens, so the run can't detect whether these edits move accuracy or 8K truncation. This tiny local run is at the accuracy floor.
- The 2048-token baseline took about 19 min per batch of 16 on this box. A full 8K public evaluation is not feasible on this CPU; it needs a GPU or the server.
- **Local ≠ official:** none of these numbers are server validation or private-test scores. The server uses vLLM with an 8192 budget and a different item set. Server scores are authoritative; I make no private-test claims.

## Honest caveats
- A is close to a duplicate of the current #1 entry. Its value is as an independent reproduction with published code, since roshaans has no PR. On its own it is unlikely to be a new #1.
- B is the novel bet, but it's untested beyond the mechanism check. Norm ×1.667 on its own scored *below* baseline (17.2), so sharpening has a dose-response risk.
- Item selection and prompts in mini_eval are ours and differ from the platform's prompt templates.

## Recommended submission
- **Primary A**, title: `Dampen Wait/Hmm token embeddings (scale 0.4) — 7-token reproduction`
- Optional second, **B**: `Wait/Hmm embed ×0.4 + final norm gain ×1.25`
- Research PR: yes, recommended, since the competition encourages it and there is no public record of the #1 recipe. Put a directory `research/joe-rlo-filler-embed-dampen/` in a fork of `ilblackdragon/deltaleague` containing this README, `make_edits.py`, `mini_eval.py`, `analyze.py`, `mini_items.json` and aggregate tables. Exclude `.safetensors` and `*.generations.jsonl`/raw jsonl, per CONTRIBUTING.md. Open the PR first, then `delta submit ... --pr URL`.

## Next steps (Joe)
```sh
cd /workspace/deltaleague-experiment/qwen35 && source ../.venv/bin/activate
delta login --no-browser                     # open URL, sign in with GitHub
# with research PR (preferred):
delta submit submission --title "Dampen Wait/Hmm token embeddings (scale 0.4) — 7-token reproduction" --pr https://github.com/ilblackdragon/deltaleague/pull/NN
# or source-free:
delta submit submission --title "Dampen Wait/Hmm token embeddings (scale 0.4) — 7-token reproduction"
# optional combo:
delta submit submission-B --title "Wait/Hmm embed x0.4 + final norm gain x1.25" [--pr URL]
delta submission status SUBMISSION_ID --watch
delta submission results SUBMISSION_ID --json
delta leaderboard --model qwen35-08b-v1
```
