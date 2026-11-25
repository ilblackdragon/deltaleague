# Commit when two of your own checks agree: a self-distilled LoRA for thinking truncation (qwen35-08b-v1)

## Hypothesis and scope

- **Experiment directory:** `research/ALuhning-consistency-commit/`
- **Attribution:** proposed, run and written up by an AI agent (Claude Code, Opus 5.5), directed, budgeted and approved by `ALuhning`.
- **What was available:**
  - the parent checkpoint;
  - `evaluation.json` and the public `validation.json`;
  - public leaderboard metadata and research PRs;
  - the **training** splits of gsm8k (`openai/gsm8k` main) and MATH (`EleutherAI/hendrycks_math`, levels 1–4).
  - No private test material. Public validation items were never used for training or example selection.
- **Parent:** `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` (Original model).
  - Content hash `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
  - Board: validation 40.29, private test 17.40.
- **Evaluation configuration:** `private1000-v6-thinking8k-20261111` (`delta-public-v1`). The private sampler is thinking on, T 1.0, top_p 0.95, top_k 20, presence penalty 1.5, 8,192 tokens.

**What the data showed first** (full trace in `TRACE.md`):
- At the private sampler the original closes its thinking within 8,192 tokens on only ~7% of held-out training-split problems, and every one of those is correct.
- The failure is open-ended re-verification, not verbatim loops. Our earlier attempts to fix it through the linear-attention memory gates (hand-set or RL) gave null results.
- **Forced-exit test.** We cut off-limit attempts at 25/50/75% and forced `</think>`. The model then answered correctly in 27–29% of gsm8k and 9–14% of MATH cases that had scored 0%.
- When two cut points produced the same answer, that answer was correct in **14 of 15** cases (92% gsm8k, 3/3 MATH).

**Hypothesis.** The model often has the answer and does not commit. If it is trained on its own attempts that stop once two of its own checks agree (answer, re-check, commit), it will close its thinking and answer within budget far more often. Accuracy should rise without new knowledge.

**Novelty check.** We searched all PR titles and bodies and all leaderboard names (107 PRs, 94 entries). The closest prior work:
- [#41](https://github.com/ilblackdragon/deltaleague/pull/41) self-distilled rare *complete* correct traces;
- [#63](https://github.com/ilblackdragon/deltaleague/pull/63) used DPO "finished beats truncated", which rescued items mostly wrongly.

Neither builds examples by cutting *failed* attempts at an agreement point.

## Reproduction

**Hardware and software.** One AMD Ryzen AI Max (Strix Halo, Radeon 8060S, ROCm):
- **Generation:** vLLM `0.33.0+strix` (`kyuz0/vllm-therock-gfx1151`), not the platform's pinned CUDA image.
- **Training:** PyTorch 2.13 ROCm, transformers 5.23, peft 0.22, `flash-linear-attention`/`fla-core` 0.6.0. The kernels were checked against the PyTorch reference.
- No rented compute.

1. **Data.** Run `prep_data.py data` (shuffle seed 20261124). The held-out set is 64 gsm8k-train + 64 MATH-train problems (`data/heldout.jsonl` sha256 `9745eabe…02a3`).
2. **Attempts to learn from.** Run `think_probe.py data/filter-candidates.jsonl … --samples 2 --cap 6144 --seed 20261125` with `--save-text`. Inputs are the first 1,024 gsm8k-train rows of the shuffle, at the private sampler.
3. **Commit examples.** Run `commit_data.py runs/filter.jsonl data/filter-candidates.jsonl data/commit.jsonl URL --max-traces 1200`.
   - Inputs: 1,200 off-limit attempts.
   - Forced exits at 20/35/50/65/80% (previous line break), answers sampled by vLLM at the private sampler (≤384 tokens), seed 20261126 + 10·trace + cut.
   - Kept where two cuts give the same boxed answer and it is correct. The example is the thinking up to the later agreeing cut, then `</think>`, then the forced answer.
   - Yield: **198** commit examples, plus **126** naturally finished correct attempts. Example ids and cut fractions are in `results/commit-ids.jsonl`.
4. **Repair examples (no yield).** `hint_data.py` half-hint rationalization on 800 MATH-train problems produced **0** usable examples (see Limitations).
5. **LoRA.** Run `train_lora_commit.py --parent model-orig --data data/commit.jsonl data/repair.jsonl --out runs/lora1 --submission runs/lora1/submission`.
   - Rank 16, alpha 32.
   - Targets: q/k/v/o, in_proj_qkv/in_proj_z/out_proj, gate/up/down in all 24 language layers (150 targets, 10.2M params).
   - lr 1e-4 cosine, 21 steps × 16 examples, completion-only loss, seed 20261126. Loss 0.72 → 0.68.
6. **Patch.** Written directly in the Delta `lora` format (fp32 A/B, alpha 32; `weights.safetensors` sha256 `0359c537…95c8`, 40,917,480 bytes; `manifest.json` sha256 `5166a11f…bc92`). `delta validate` passes. The evaluated checkpoint was rebuilt from the patch with `delta_cli.artifacts.reconstruct` (`reconstruct_eval.py`).
7. **Held-out check.** Run `think_probe.py data/heldout.jsonl … --samples 1` (seeds 20261124 + 1000·item) and `heldout_compare.py`.

The orchestration scripts are `run_commit.sh` and `run_suite.sh`; hashes are in `results/hashes*.txt`.

## Evidence (local, advisory; official scores arrive only after submission)

**Held-out problems at the private sampler** (128 training-split problems not used for training, same seeds for both models, one sample each):

| | original → candidate correct | thinking closed | cut off at 8,192 | paired diff (95% bootstrap CI) | wins / losses |
|---|---|---|---|---|---|
| gsm8k (64) | 5 → **29** | 5 → 35 | 59 → 35 | **+37.5** [+26.6, +50.0] | 24 / 0 |
| MATH (64) | 3 → **14** | 5 → 16 | 61 → 50 | **+17.2** [+7.8, +28.1] | 12 / 1 |
| all (128) | 8 → **43** | 10 → 51 | 120 → 85 | **+27.3** [+19.5, +35.2] | 36 / 1 |

- Mean tokens fell from 7,994 to 6,915.
- No answer that closed and completed was wrong.
- MATH improved although the LoRA saw no MATH examples.

**Public validation suite** (pinned `delta public-eval run` with the config's sampler):
- Both models completed the same **8 of 18** benchmarks (502 of 1,110 items).
- Both then failed identically entering math500 with an `AssertionError` in this local runtime, so `delta compare` could not run.
- These are paired local scores, single sample each, roughly ±6 points per benchmark:

| bbh | gsm8k | commonsense_qa | arc_challenge | boolq | arc_easy | hellaswag | ifeval | mean of 8 |
|---|---|---|---|---|---|---|---|---|
| 7.4 → 27.8 | 43.8 → 53.1 | 54.7 → 57.8 | 31.2 → 32.8 | 65.6 → 65.6 | 64.1 → 62.5 | 46.9 → 45.3 | **68.8 → 59.4** | 47.8 → 50.5 |

The **ifeval drop (−9.4, 6 of 64 prompts)** is a possible regression in instruction following, and it is reported as one.

## Limitations and negative findings

- **The held-out problems are not public validation.** They come from the same training-split distributions as the commit data (gsm8k train); MATH is out of domain for the training data. There is one sample per problem.
- **10 public benchmarks were not evaluated locally** (math500, mmlu, mmlu_pro, openbookqa, piqa, winogrande, humaneval, mbpp, bfcl single/multi) because of the local evaluator failure. Effects on coding and tool use are unknown.
- **The repair arm failed.** With a half-solution hint, the original still exceeded 4,096 tokens on 797/800 problems, and 461 attempts mentioned the hint. Repair data has to come from a model that already commits (a second STaR round), which was not run.
- **Earlier null results in the same trace:**
  - a hand-set memory edit (A_log x0.5 on 108 heads);
  - gate-only GRPO (3 updates; held-out 8/128 vs 8/128).
  - We also dropped a temperature edit before running it, because #7 and #42 already cover it.
- **The served runtime differs from the platform's.** That is ROCm vLLM vs the pinned CUDA image; the sampler settings match.

## Submission

- Submission: [`b94b64fe-1aac-4394-b7fa-6a7be1d3f8b0`](https://deltaleague.ai/submissions/b94b64fe-1aac-4394-b7fa-6a7be1d3f8b0/), version `15049dff-7d2c-4537-aef1-e6f324a0ebb1`, parent `0ea62f40` (Original model), config `private1000-v6-thinking8k-20261111`.
- **Verified (platform) scores, same evaluation config as the original:**

| | original | this submission | diff |
|---|---|---|---|
| private test | 17.40 ± 1.20 | **20.40 ± 1.27** | +3.0 |
| validation (18 benchmarks) | 40.29 | **41.99** | +1.70 |
| leaderboard rank | 91 | 89 (of 97) | |

- The +3.0 private-test gain is below the leaderboard's independence-based band (1.96·√(1.27² + 1.20²) ≈ 3.4), so it is **not established** as a separation from the original.
- Verified validation by benchmark (each benchmark 32–114 items):

| benchmark | original | submission | diff |
|---|---|---|---|
| arc_challenge | 32.8 | 34.4 | +1.6 |
| arc_easy | 64.1 | 62.5 | -1.6 |
| bbh | 22.2 | 16.7 | -5.6 |
| bfcl_multi_turn | 3.1 | 0.0 | -3.1 |
| bfcl_single_turn | 48.6 | 55.6 | +6.9 |
| boolq | 65.6 | 64.1 | -1.6 |
| commonsense_qa | 57.8 | 57.8 | +0.0 |
| gsm8k | 51.6 | 59.4 | +7.8 |
| hellaswag | 46.9 | 46.9 | +0.0 |
| humaneval | 15.6 | 15.6 | +0.0 |
| ifeval | 57.8 | 62.5 | +4.7 |
| math500 | 20.3 | 34.4 | +14.1 |
| mbpp | 9.4 | 12.5 | +3.1 |
| mmlu | 47.4 | 45.6 | -1.8 |
| mmlu_pro | 25.7 | 28.6 | +2.9 |
| openbookqa | 29.7 | 31.2 | +1.6 |
| piqa | 70.3 | 70.3 | +0.0 |
| winogrande | 56.2 | 57.8 | +1.6 |

**Local vs verified.**
- The local held-out estimate (+27.3 points on gsm8k/MATH training-split problems) greatly overstated the private-test effect (+3.0).
- On the platform the gains concentrate where the held-out set lived: math500 +14.1, gsm8k +7.8, bfcl single-turn +6.9.
- Most other benchmarks moved within noise; bbh and bfcl multi-turn fell.
- The local ifeval drop (−9.4) did not reproduce: verified ifeval rose +4.7, so it was likely noise.
- The private test mixes many task types, and the commit habit was learned from gsm8k only.

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs (results hold ids, seeds, flags and aggregate scores only; training text is not included).
- [x] Local estimates and verified results are clearly distinguished.
