# GRPO with a gentle length penalty on fb984a19 (step 8), plus a negative 9B-distillation result

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, rank 1, private test 53.4).

## Hypothesis

About 14% of `fb984a19`'s private-test answers are truncated. On held-out MATH train items, a third of its answers hit the 8k cap.

In #69, 8 plain GRPO steps on `fb984a19` with roshaans' recipe (#58) scored 51.5 privately. Its local signals were flat to negative, and truncation did not fall. This run keeps the recipe, but rewards **shorter correct answers** a little more, so the policy gradient has a reason to stop rambling. A strong cap is avoided: #58 found that a 4k training cap made the model rush.

**Reward change** (`grpo-len-shape.diff`, a 4-line patch to #58's `grpo.py`): a correct, finished answer longer than 3,000 tokens loses reward linearly. At the 6,144-token training cap it has lost half: `r * (1 − 0.5 · min(1, (len − 3000) / 3144))`. Wrong or truncated answers still score 0.

A side effect: groups whose samples were all correct now carry an advantage signal. Step 1 trained on 192 sequences, compared with 152 without the penalty on the identical rollouts.

## Method

- **Settings:** #58's `grpo.py` with `--mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 --train-max-tokens 6144 --match-sampler --len-shape 3000`, 16 steps, with checkpoints at steps 8 and 16. Reward-decline stop disabled: in #69 it fired on batch noise.
- **Data and compute:** 64 prompts × 8 samples per step, lr 1e-6. Public train splits only, with the probe items excluded. One rented RTX 6000 Ada, about 3 min per step. Driver: `run_c.sh`. Training log: `results/train_log_C.jsonl`.
- **Patch:** `delta patch build` against `fb984a19` for step 8 changes 253 tensors, all under `model.language_model`. `delta validate` passes.

| Artifact | sha256 |
|---|---|
| step-8 weights | `f68e3c1196d80b8d3fecdff5f958e32cd37b7b1676f8d07a77e9151b9b72fd9a` |
| patch weights | `c59ef7b050fbc7c1562ca398415d56233e01c1403a3ce6b37c2bdd2a290923ee` |
| patch manifest | `6d4fe1296b6670ea3b7f740b74dcce299f4d1b19201c52b2d7443e26b99eb46b` |

## Local evidence (advisory)

Both measurements reuse #69's harness. The parent's numbers come from #69's pod: same GPU type, same seeds and same items, but a different machine.

**Math probe:** 256 gsm8k and 256 MATH held-out train items, evaluation sampler, 8k cap.

| | `fb984a19` | **step 8** | step 16 |
|---|---|---|---|
| gsm8k | 75.4 | 75.4 | 73.0 |
| MATH | 40.2 | 46.5 | 45.3 |
| all | 57.8 | **60.9** | 59.2 |
| truncated | 17.0% | 14.1% | 11.5% |
| median tokens | 1,892 | 1,782 | 1,606 |
| paired vs parent | | **+3.12 ± 1.65** (44 wins, 28 losses) | +1.37 ± 1.86 |

**Public suite:** 14 text groups and 2 coding groups, BFCL excluded, thinking on, 8k cap.

| Group | `fb984a19` | **step 8** | step 16 |
|---|---:|---:|---:|
| bbh | 33.3 | 35.2 | 44.4 |
| gsm8k | 62.5 | 67.2 | 54.7 |
| ifeval | 71.9 | 76.6 | 75.0 |
| math500 | 59.4 | 59.4 | 50.0 |
| mmlu_pro | 32.9 | 35.7 | 32.9 |
| mmlu | 48.2 | 49.1 | 49.1 |
| humaneval | 28.1 | 25.0 | 28.1 |
| mbpp | 25.0 | 21.9 | 31.3 |
| winogrande | 60.9 | 60.9 | 62.5 |
| other groups | — | within ±1.6 | within ±1.6 |
| **mean (16)** | **49.05** | **49.40** | **49.41** |

**Reading:**

- **Step 8** is the first continuation of `fb984a19` in our runs that beats it on the probe by about 1.9 SE. It gains on gsm8k, IFEval and MMLU-Pro and loses a little on code.
- **Step 16** is shorter, but its gains are uneven: BBH +11, while gsm8k −8 and MATH-500 −9. It looks over-compressed.
- The public-suite gain is within noise.

## Negative result: 9B-teacher distillation (same pod, same day)

Before this run, we tried distilling `Qwen/Qwen3.5-9B` into `fb984a19`. Scripts: `run_distill.sh`, `gen_teacher.py`, `select_traces.py` and `train_sft.py`.

**Setup:**

- The teacher answered 4,000 train prompts twice, using the evaluation sampler with a 6k cap. Its accuracy was about 65%, and 18% of its answers were truncated.
- For each prompt we kept the shortest correct trace that closes `</think>` once and is at most 4,096 tokens long. That gave 2,407 traces averaging 1,828 tokens.
- We then ran full SFT for 1 epoch at lr 5e-6.

**Result:** a clear loss.

- Math probe: 47.5 (truncation 37.7%, median 4,864 tokens).
- Public mean: 39.7. BBH fell to 5.6, IFEval to 51.6, gsm8k to 45.3 and MBPP to 6.3.
- The student copied the teacher's long, exploratory style without the capacity to finish, even though it was trained only on short traces.
- See `results/*_sft.*` and `results/teacher_selection.txt`. A half-strength blend was not evaluated.

## Credit

Credit goes to:

- roshaans: `grpo.py`, the recipe and the probe (#58);
- SpacexCeo: `fb984a19`;
- dawgster, ma4ypic4y and the #43, #55 and #62 lineage.
