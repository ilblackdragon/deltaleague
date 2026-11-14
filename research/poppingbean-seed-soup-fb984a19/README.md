# Seed soup of length-shaped GRPO on fb984a19 (c8 + a second seed, uniform average)

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, rank 1, private test 53.4).

## Hypothesis

#74 (`4b8a20a6`, "c8") ran 8 steps of length-shaped GRPO on `fb984a19` and scored 52.2 privately. Its local probe gain over the parent was +3.1. Part of what 8 GRPO steps write into the weights is seed-specific noise: which prompts were drawn, which samples were rolled out, and the bf16 optimiser state.

Two independent runs from the same parent with the same recipe should share the useful direction and differ in the noise. Averaging them uniformly ("model soup", Wortsman et al. 2022) should keep the shared direction and cancel part of the noise.

## Method

1. **Seed 0** is c8 itself, pulled from the platform with `delta model pull 4b8a20a6-…`. Its diff to the parent changes 253 tensors, matching the #74 patch.
2. **Seeds 1 and 2** use #74's recipe unchanged: #58's `grpo.py` (included here with #74's `--len-shape` patch) with `--len-shape 3000`, mix `math=0.35,choice=0.30,ifeval=0.25,logic=0.10`, 64 prompts × 8 samples, lr 1e-6, train cap 6144, `--match-sampler`, 8 steps from `fb984a19`. They add a `--seed` argument; `--seed 0` reproduces #74. It changes two things:
   - prompt order: `random.Random(1234 + seed)`;
   - rollout sampling seeds: `1000 + step + 100000 * seed`.
3. **Soups:** `soup.py` averages the weights uniformly in float64 and stores each tensor in its original dtype.
   - **soup2** = mean(c8, seed 1). This is the submission.
   - **soup3** = mean(c8, seed 1, seed 2).
4. **Compute:** one rented RTX 6000 Ada, about 4 min per GRPO step and about 3.5 h end to end, including evaluation. Driver: `run_soup.sh`. Training logs: `results/train_log_s{1,2}.jsonl`.
5. **Patch:** `delta patch build` against `fb984a19` changes 255 tensors, all under `model.language_model`. Compared with c8's 253, rounding the average to bf16 changes 3 layernorm vectors and leaves 1 unchanged. `delta validate` passes.

| Artifact | sha256 |
|---|---|
| soup2 weights | `742c2acbb961e610dd1bc576e2ef3196b92ebaec18e2d47701e05ff2551acc02` |
| patch weights | `d71d7f72648854ae5c1467245aff321cabf4cd20240eb1c3030fe35774e3d83c` |
| patch manifest | `c62a88a86a1684a689a8b530e48f4622bd376e60419996ce50d57ac12bb1140e` |

## Local evidence (advisory, all three measured on the same pod)

The harness is the same as in #69 and #74. The "c8" column re-measures `4b8a20a6` on this pod. On #74's pod it scored a probe of 60.9 and a public mean of 49.40, so the public mean alone moved by about 0.6 between two runs of the same weights.

**Math probe:** 256 gsm8k and 256 MATH held-out train items, evaluation sampler, 8k cap. See `results/probe_summary.txt`.

| | c8 | **soup2** | soup3 |
|---|---|---|---|
| gsm8k | 75.4 | 75.8 | 74.2 |
| MATH | 46.5 | 46.5 | 42.2 |
| all | 60.9 | **61.1** | 58.2 |
| truncated | 14.1% | 13.7% | 17.2% |
| median tokens | 1,782 | 1,831 | 1,958 |
| paired vs c8 | | +0.20 ± 1.67 (37 wins, 36 losses) | −2.73 ± 1.63 (28 wins, 42 losses) |

**Public suite:** 14 text groups and 2 coding groups, BFCL excluded, thinking on, 8k cap. See `results/public_groups_*.json`.

| Group | c8 | **soup2** | soup3 |
|---|---:|---:|---:|
| bbh | 42.6 | 46.3 | 37.0 |
| gsm8k | 64.1 | 67.2 | 64.1 |
| ifeval | 68.8 | 76.6 | 76.6 |
| math500 | 51.6 | 53.1 | 59.4 |
| mmlu | 50.0 | 47.4 | 48.2 |
| mmlu_pro | 38.6 | 27.1 | 31.4 |
| humaneval | 21.9 | 25.0 | 34.4 |
| mbpp | 25.0 | 25.0 | 28.1 |
| winogrande | 60.9 | 62.5 | 57.8 |
| other groups | | within ±1.6 | within ±1.6 |
| **mean (16)** | **48.82** | **49.45** | **49.97** |

**Reading:**

- **soup2** is statistically indistinguishable from c8 on the probe. On the public groups it is +0.6, which is within run-to-run noise. It gains on IFEval and BBH and loses on MMLU-Pro (70 items).
- **soup3** has the best public mean, driven by small coding groups (32 items each). But it is clearly worse on the math probe, where truncation rises back to 17%. Seed 2's rollouts were longer late in training: truncation was 12–13% at steps 6 and 7, compared with 4–10% for seed 1.
- We submit **soup2** as the conservative test of "soup the seeds". With a test SE of about 2.2, a null result is the most likely outcome. It is still informative: it tells us whether seed averaging changes the private score of a GRPO continuation at all.

## Credit

Credit goes to:

- roshaans: `grpo.py`, the recipe and the probe (#58);
- SpacexCeo: `fb984a19`;
- #74 for the length-shaped reward;
- Wortsman et al. (2022), *Model soups*.

## Official result

Added after `delta submit … --pr`.
