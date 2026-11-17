# Fixing the JSON-files coding contract on fb984a19 by a small supervised fine-tune on the model's own code

Research by natgurlain with an AI agent (Claude Code). Available during research: public leaderboard rows, public checkpoints, public research PRs, public training splits of open datasets, the `delta` CLI and its public scoring code. No private test material.

- **Parent:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, private 53.4), content hash `bd3c11a04a4d1691c1de6b7afadd18d8a56c232d29aa8959fec41819ff43ddfd`
- **Built on:** my #82 (`fb984a19` + 60 steps Fair Cast GRPO, verified private 52.9). The patch here is relative to `fb984a19` and contains #82's change plus this one.
- **Evaluation configuration:** `private1000-v6-thinking8k-20261111` (`delta-public-v1`)
- **Submission:** not submitted yet

All statements about quality below are local and advisory.

## Hypothesis

The CLI's custom-task scorer has a coding track: the prompt is wrapped by `coding_prompt(task, starter_files)` and the answer must be either a JSON object `{"files": {"solution.py": "..."}}` or bare valid Python (`candidate_files`). I measured how this lineage copes with that contract on open MBPP test tasks (a measurement-only census, no training):

| Model | Valid response | Passes the tests | Same tasks, plain code-block format |
|---|---:|---:|---:|
| parent `fb984a19` | 46.5% | 13.2% | 26.7% |
| #82 (Fair Cast step 60) | 52.8% | 15.2% | 36.7% |

About half of the answers are rejected before any test runs. The failures are serialisation slips, not wrong code: Python triple quotes inside the JSON string, raw newlines or text after the object, a missing closing brace. The same model passes the same tasks about twice as often when it may answer in a code block.

Hypothesis: a short supervised fine-tune that shows the model **its own code, serialised correctly**, repairs the contract without touching what the model knows, and without harming other answer formats.

Whether the private suite contains such tasks is not known to me. This experiment is about the capability, measured on open data.

## Method

Base: #82's step-60 checkpoint. Tasks: MBPP `train` + `validation` + `prompt` splits only (474 tasks). The 500 MBPP test tasks are never trained on; 150 of them are the test slice.

1. **Prompts** (`jsonfix_build.py`): 414 training tasks and 60 dev tasks wrapped by the CLI's own `coding_prompt`. Starter-file variants on the training tasks: empty `solution.py` (294), a stub `solution.py` (60), another file name (60).
2. **Own samples** (`jsonfix_gen.sh`): the base model answers each prompt 4 times with the evaluation sampler (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, 8,192 tokens, thinking on).
3. **Rewrite** (`jsonfix_data.py`): for every sample whose thinking closed, recover the code from the visible answer (accepted as it is, or repaired: triple quotes, raw newlines or trailing text, missing brace, trailing comma, code block) and rewrite the visible answer as `json.dumps({"files": {name: code}})`. The code is the model's own; no reference solutions and no tests are used to filter, so wrong code stays wrong. 49% of samples were already valid, 28% were repaired, 19% were unrecoverable and dropped, 3% unfinished. Result: 1,288 JSON sequences (816 already valid, 472 repaired), every target accepted by `candidate_files`.
4. **Replay** (anchor): 858 of the base model's own finished answers in other formats, kept verbatim: the same tasks in the plain code-block format, pool prompts (boxed math, multiple choice, instruction following, logic) and answer-only prompts.
5. **Fine-tune** (`sft_json.py`): on JSON sequences the loss is cross-entropy on the visible answer only (the sampled thinking is context, with the presence penalty applied to the logits); on replay sequences the loss is the negative log-probability under the evaluation sampler, which has zero expected gradient at the start and so only resists drift. fp32 master, AdamW (0.9, 0.99), learning rate 2e-6, 10 warm-up steps, 150 steps, 5 repaired + 5 valid + 6 replay sequences per step, gradient clip 1.0, bf16 export by stochastic rounding (as in #82) at steps 50, 100, 150, seed 20261117.

All settings, the checkpoint-selection rule and the pass/fail thresholds were written down before any generation or training.

## Evidence

Everything is paired with the base (#82 step 60) on the same prompts, 4 samples per prompt, evaluation sampler. Scoring uses the CLI's own `final_answer`, `score_answer` and `candidate_files`, and the WASI sandbox for tests (`census_score.py`; it reproduces the harness's plain-format MBPP score exactly).

**Dev set** (60 held-out training-split tasks x 4; `results/json_coding_special_casing.txt`):

| Checkpoint | Valid | Passes tests | Hit the token cap |
|---|---:|---:|---:|
| base | 52.5% | 21.2% | 2.5% |
| step 50 | 91.7% | 24.2% | 2.5% |
| step 100 | 95.8% | 33.3% | 2.1% |
| step 150 | 92.1% | 41.7% | 5.0% |

**Step 50 — first test, failed the rule.** The rule fixed in advance picked the earliest checkpoint with dev valid rate of at least 90%, which is step 50. On the test slice it reached valid 91.0% but pass rate only 18.7% against a threshold of 22% (`results/census_fc60_step50.txt`). All guards passed. By the rule it was not a candidate. The pass rate among valid answers had fallen from 28.7% to 20.5%: the newly valid answers carried weaker code.

**Step 150 — second look at the same test slice.** Because the dev pass rate kept rising, step 150 was then tested. This is a second look at the same 150 tasks with a checkpoint chosen after seeing the dev table and the step-50 result. The thresholds were therefore not relaxed, two were added (rows marked "added"), and no further checkpoint was tested. Predictions and thresholds were written down before generation.

| Readout | Base | Step 150 | Difference | Threshold | Verdict |
|---|---:|---:|---:|---|---|
| JSON coding, valid response (150 test tasks x 4) | 52.8% | 94.2% | +41.3 ± 2.5 | at least 90% | pass |
| JSON coding, passes tests | 15.2% | 27.8% | +12.7 ± 2.3 | at least 22% | pass |
| added: passes tests, excluding answers that special-case the shown tests | 13.3% | 23.5% | +10.2 ± 2.1 | at least 20% | pass |
| added guard: JSON answers hitting the token cap | 4.7% | 4.3% | | at most 8% | pass |
| guard: answer-only prompts, exact match (300 prompts x 4) | 64.2% | 67.0% | +2.8 ± 1.25 | not below −3.5 | pass |
| guard: truncation-prone boxed math (150 prompts x 4) | 14.8% | 17.5% | +2.7 ± 1.6 | not below −4 | pass |
| guard: MBPP plain format, 500 test tasks, public harness, 1 sample | 31.6% | 32.8% | +1.2 [−2.8, +5.2] | not below −5 | pass |
| guard: BFCL single turn, 366 items, public harness, 1 sample | 68.6% | 71.6% | +3.0 (unpaired SE 3.3) | not below −8 | pass |
| guard: 4-family probe (500 held-out prompts x 4) | 67.90 | 66.90 | −1.00 [−3.00, +1.00] | not below −3.0 | pass |

Files: `results/census_fc60_step150.txt`, `results/tool_calling.txt`, `results/mbpp_plain_format.txt`, `results/cmp_jf_s150_p2h_vs_fc60.txt`.

The pass rate among valid answers is back at the base's level (28.7% base, 20.5% step 50, 29.6% step 150).

Against the parent `fb984a19` on the same slices: valid +47.7 ± 2.2, pass rate +14.7 ± 2.5, pass rate excluding special-cased answers +10.8 ± 2.3, answer-only exact match +2.5 ± 1.5, truncation-prone math +8.3 ± 1.8, 4-family probe +1.60 [−0.65, +3.80] (`results/cmp_step150_vs_parent.txt`; #82 on the same 500 prompts: +2.60 [+0.40, +4.75]).

**Training log** (`results/train_log.jsonl`): losses are nearly flat. Means over steps 1–25 and 101–150: repaired 0.136 → 0.101, already valid 0.165 → 0.168, replay 0.346 → 0.341.

## Reproduction

```sh
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output exp-leader
# runs/fc/step60 is the checkpoint of #82; mined/ and probe/ files come from #81 and #82
python census_build.py            # ../census/items.jsonl: 300 answer-only, 150 JSON coding, 150 truncation-prone prompts
./census.sh                       # measurement only: parent and base on the census, BFCL shards through the public harness
python jsonfix_build.py           # prompts
./jsonfix_gen.sh                  # the base samples its own answers (4 per prompt)
python jsonfix_data.py ../jsonfix/gen_fc60.jsonl ../jsonfix/replay_fc60.jsonl --out ../jsonfix/train.jsonl
python sft_json.py runs/fc/step60 - --data ../jsonfix/train.jsonl --check    # tokenisation and loss mask, CPU
./jsonfix_train.sh                # 150 steps, checkpoints 50/100/150, dev samples
python jsonfix_dev.py ../jsonfix/dev_s50.jsonl ../jsonfix/dev_s100.jsonl ../jsonfix/dev_s150.jsonl
./jsonfix_test.sh 150             # census slices, MBPP plain and BFCL through the public harness, probe
python census_score.py fc_s60 jf_s150
python jsonfix_special.py census leader fc_s60 jf_s50 jf_s150
python bfcl_cmp.py fc_s60 jf_s150 && python broad_cmp.py ../broad/fc_s60 ../broad/jf_s150
delta patch build --parent exp-leader/model --modified runs/jsonfix/step150/model.safetensors --output submission
delta validate submission
```

`census_build.py` reads the MBPP test tasks from a saved public-harness run of the parent (`delta public-eval benchmark` with the `mbpp` benchmark, full coverage). `mine.py` and `grpo.py` are the versions used here (`mine.py --save-text` keeps the generated text locally; none of it is in this record).

Environment: RTX 3090 (24 GB), Python 3.12, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, `deltaleague` 0.3.0. GPU time: census 0.9 h, own samples 0.4 h, training and dev samples 0.8 h, each test with all guards 0.85–0.9 h. vLLM sampling on a shared GPU is not bit-reproducible, so a rerun will give different weights.

**sha256:**

- parent `model.safetensors`: `fc2a115f0507251baebef840d727b3bbf626ee861e8ed4d6bd33c0e7f81441e6`
- base (#82 step 60) `model.safetensors`: `64e6b3833e2f5dd68733be1c272a4d20f295177b349f021da098ca8936ff25aa`
- step 150 `model.safetensors`: `c7652804e1dd6efa2fb6d45b14a4558f6c73988beba29ffe238df57202b7f800`
- patch `weights.safetensors` (sparse, 304 tensors): `9a9790882a132d0178a3cda1f66115581b94707c9d4a15e4aea34d83d87db0c6`
- `manifest.json`: `cb9581caf394400a6eaae24b29961336198af6a9f4d0f2f97271fbc4beaa5e8d`
- census prompt list `items.jsonl`: `26fee4e3cc1038c44c5c2232cd6c12fb24b9372f0bd9ae19dde5e4f1b4e918b7` (built by `census_build.py`)

`delta validate` passes.

## Limitations and negative findings

- **The link to the private test is unknown.** I do not know whether the private suite uses the JSON-files coding contract. If it does not, this change is expected to be neutral at best. #82 was +2.2 on my local probe and −0.5 on the private test; my local probes have not yet predicted a private result.
- **Step 50 failed its pre-set rule; step 150 is a second look** at the same 150 test tasks. The pass-rate gain is about 5.5 standard errors, so two looks do not explain it, but the checkpoint was chosen with knowledge of the first result.
- **Answers that special-case the shown tests.** MBPP prompts show their tests, and some passing answers return the expected values as literals (`if n == 10: return 325`). Share of passes: parent 3.8%, base 12.1%, step 50 5.4%, step 150 15.6% (`jsonfix_special.py`, a simple textual detector). The fine-tune trains on the model's own code without filtering, so it does not remove this habit. The gain excluding such answers is +10.2 ± 2.1. Tasks with hidden tests would not reward them.
- **4-family probe points slightly down** against the base (−1.00 [−3.00, +1.00]; instruction following −2.4 ± 2.1, math −1.5 ± 1.7, multiple choice −0.8 ± 1.8, logic +3.2 ± 3.0). Inside noise and inside the guard; step 50 showed −0.15.
- **Tool calling by category is noise-sized** (70 items, one sample each): irrelevance +17.1, parallel +8.6, multiple −7.1. Only the total is used.
- One training run, one seed. MBPP only: other languages, multi-file tasks and the terminal track were not measured.
- Earlier negative results on this lineage that led here (measured on held-out prompts, no private data): 60 more Fair Cast GRPO steps after #82 erased its local gain (−3.05 against #82), largely because "fewer than N words" instructions went from 81% to 5% compliance; GRPO's gains on this parent came from finishing more answers, not from higher accuracy among finished ones (70% → 71% → 68%); the answer-only exact-match format was already fine (0.8% of samples knew the answer but failed the format).
