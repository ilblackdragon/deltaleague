# Sampling gain on the leader: what responds, what does not, and a last-layer edit that sets it per task type (CPU only)

Status: submitted and evaluated. The entry scored 53.5 on the test (standard error 1.6)
against 54.3 for its parent, which is no measurable difference (see Submission). Everything
else said here about the candidate is a local estimate.

## Hypothesis and scope

- Experiment directory: `research/Sun-Jc-state-dependent-gain/`
- Parents: sections 1 to 7 were run on `fb984a19-6262-48db-bd14-a5d68e246ea6` (leader on
  2026-11-13, test 53.4, truncation 14.2%). The patch is the same edit, fitted again on
  `eff1e1b7-18ef-4c77-83ef-25fdcc027e75` (leader on 2026-11-15, test 54.3, truncation 14.2%),
  which differs from `fb984a19` only in two layer-23 neurons that the edit does not touch
  (section 9). Evaluation configuration `private1000-v6-thinking8k-20261111`.
- Attribution: experiments were designed, run and written up by Claude (Anthropic's coding
  agent, Claude Code) on the contributor's machine at the contributor's request. The
  contributor approved this PR and the submission.
- Information available: this repository, the public leaderboard (`delta leaderboard`),
  public checkpoints (`delta competition init --parent`), the READMEs of open research PRs,
  and public datasets (train splits of GSM8K, MATH, ARC-Challenge, CommonsenseQA, MMLU
  auxiliary train; IFEval, used for measurement only). No private test material was accessed
  or sought.
- Compute: one macOS VM, 12 vCPU, no GPU. About 170 CPU-core-hours, nearly all of it sampling.

Hypotheses, in the order they were tested:

1. *The remaining truncation (14.2% on the test) is recoverable.* **Refuted locally**: the
   best-case decode-time exit recovers 2 of 38 capped completions (section 2).
2. *Score is lost while the answer is written.* **Refuted locally**: with the thinking held
   fixed, re-sampling the answer changes the outcome for 6 of 59 prompts (section 4).
3. *The final-norm gain still has slope on this parent.* **Not as a global step**: on the
   reasoning families ×1.2 costs 5.0 ± 2.4 points and ×0.85 costs 2.8 ± 2.5. Instruction
   following does not follow that pattern (section 3).
4. *The gain can be made to depend on the model's state with a small last-layer edit.*
   **Yes**: residual coordinate 0 is softmax-neutral, and writing into it rescales the logits
   without changing their pattern. Three layer-23 neurons plus the final norm give a gain of
   1.2 on prose-writing tasks and leave reasoning states at 1.00 ± 0.02 (sections 5 and 6).
5. *Instruction following gains from a sharper sampler.* **Not established**: the edited
   model scores 0.0 ± 3.5 points against the parent on 136 fresh IFEval prompts and
   +1.5 ± 3.0 on all 200 (section 7).

The net result is negative for score and positive for tooling: four natural levers on this
parent are measured and found flat, and the candidate is a neutral edit that demonstrates the
mechanism of point 4.

Lineage and credit: the patch's parent is Congoss's `eff1e1b7` (#80), a 25% rollback of the
#66 graft neurons on SpacexCeo's `fb984a19`, which stands on dawgster's `fdad0374` (#66),
poppingbean's `ac147fa6` (#62), roshaans's `0c834b41` delta (#58), ma4ypic4y's `9492e219` and
earlier entries. The edit here reuses two ideas from dawgster's #66: a pair of SwiGLU neurons
whose difference is a soft step on a linear probe, and reading that probe at the layer-23 MLP
input. The split of truncation into "never closed" and "closed, then overflowed" follows #49,
#53 and #65. Final-norm steps follow zavodil (#56, #60), poppingbean (#43) and earlier
entries.

## Method

Everything runs on CPU with a batched re-implementation of the evaluation sampler
(`engine.py`): presence penalty 1.5 on generated tokens, then top-k 20, then top-p 0.95,
temperature 1.0, cap 8,192 tokens, thinking on. `test_engine.py` checks it against stock
`transformers` (worst logit deviation 1.3e-4 in float32). Weights are the BF16 checkpoint
cast to float32; the platform computes in BF16, so individual completions differ, but the
sampling distribution is the same up to BF16 noise.

- **Prompts** (`data.py`, `make_jobs.py`): 384 prompts from public train splits in the
  format of `benchmarks/heldout-v1` (`…\nPlease reason step by step, and put your final
  answer within \boxed{}.`): 120 MATH (levels 2–5), 80 GSM8K, 48 ARC-Challenge, 36
  CommonsenseQA, 36 MMLU auxiliary-train, plus 64 IFEval prompts. One sample each, seed 1000.
  "Reasoning families" below means the first five (320 prompts).
- **Scoring** (`scorelib.py`, `analyze.py`, `compare_runs.py`): `delta_cli.scoring.score_answer`,
  unchanged. Two local adaptations, both described under "Scorer observations": math golds are
  wrapped as `$…$`, and langdetect is seeded.
- **Gain emulation** (`gen.py --gain G`): scaling the final-norm gain by G multiplies every
  logit by G, which under the fixed sampler equals temperature 1/G with presence penalty
  1.5/G on the unscaled logits. No checkpoint is needed for a dose-response.
- **Replay oracles** (`oracle.py`, `answer_gain.py`): a stored completion is replayed with the
  same per-sequence random stream and something is changed from a chosen step on: tokens are
  forced, logits pushed, or the answer re-sampled at another gain. The result is paired with
  the original completion, so the difference is caused by the intervention alone. This bounds
  what a weight edit acting at that point could achieve, before any edit is built.
- **Last-layer edits offline** (`acts.py`, `phase_gain.py`): the layer-23 MLP feeds only the
  final norm and the output embedding. With its input and the final residual stored for a set
  of states, an edit confined to that MLP and the final norm is evaluated exactly on those
  states without running the model. Fits are always evaluated on held-out completions.
- **Verification on the real forward pass** (`verify_gain.py`, `edit_eval.py`): stored
  completions are teacher-forced through the parent and the edited model; at every position
  the edited logits are regressed on the parent's (slope = effective gain) and the two
  sampling distributions are compared (total variation after presence penalty, top-k, top-p).

## Evidence

All numbers in this section are **local estimates** on the prompts above. They are not
leaderboard results. With one sample per prompt the standard error of a score on 384 prompts
is about 2.4 points; paired differences are given with their own standard errors.

### 1. Where the leader loses

| Family | n | Score | Finished | Cap while thinking | Cap after closing | Score when finished | Mean tokens |
|---|---|---|---|---|---|---|---|
| All | 384 | 64.6 | 90.1% | 3.6% | 6.2% | 70.8 | 2,483 |
| MATH | 120 | 50.0 | 71.7% | 9.2% | 19.2% | 67.4 | 4,949 |
| GSM8K | 80 | 78.8 | 100% | 0 | 0 | 78.8 | 1,468 |
| ARC | 48 | 83.3 | 100% | 0 | 0 | 83.3 | 960 |
| CommonsenseQA | 36 | 55.6 | 100% | 0 | 0 | 55.6 | 576 |
| MMLU | 36 | 72.2 | 100% | 0 | 0 | 72.2 | 955 |
| IFEval | 64 | 60.9 | 93.8% | 4.7% | 1.6% | 63.3 | 2,203 |

- 38 completions hit the cap (9.9%); they consume 33% of all generated tokens. 24 of them had
  already emitted `</think>` and ran out of budget in the answer; 14 never closed.
- Every finished MATH or GSM8K answer that scores 0 (45 of them) has one `\boxed{}` with a
  wrong value. None is a formatting loss.
- Capped traces are not verbatim loops. They re-derive and second-guess.

### 2. Exit oracles: capped completions are not recoverable (negative)

Each row replays capped completions with a rule and scores the result against the original.
"Before" counts capped completions that score anyway, because `math-verify` falls back to
the last expression of an unfinished text.

| Rule | Replayed | Finish now | Correct before → after | Net on 384 prompts |
|---|---|---|---|---|
| Force `</think>` at step 7,800, then `\n\n\` after every `</think>` | 38 | 25 | 3 → 5 | +0.52 |
| Push `</think>` by +20 logits from step 6,000, same boxing rule | 38 | 16 | 3 → 3 | 0.00 |
| Box right after the model's own first close at step ≥ 3,000 | 19 | 19 | 8 → 8 | 0.00 |
| … at step ≥ 4,500 | 14 | 14 | 1 → 1 | 0.00 |
| … at step ≥ 6,000 | 9 | 8 | 2 → 1 | −0.26 |
| Box right after the first close, 185 finished completions replayed | 185 | 185 | 138 → 135 | −0.78 |

- A forced exit makes most capped completions end cleanly, but the forced answer is right
  about as often as the unfinished text already scores by accident. The problems on which
  the leader runs out of budget are problems it cannot solve. This agrees with #53 and puts
  a ceiling of roughly one point on any deadline-style edit on this parent.
- Observations that a deadline edit would have to deal with, recorded because they are
  reusable:
  - The model keeps its own thinking state through a forced `</think>`: after a forced close
    and a boxed answer it emitted `</think>` again in 15 of 38 replays, then started a
    write-up that the cap cut off. The layer-23 "ready" graft neurons from #66 (1998, 3529)
    are not the cause.
  - The number of generated tokens is linearly decodable from the residual stream (held-out
    R² 0.85 after layer 6). It comes from slow linear-attention heads: layer 2 head 11 alone
    gives R² 0.85, layer 6 head 8 gives 0.83. The signal saturates near 5,000 tokens.
  - A token direction written into the residual stream moves its final logit by about 14 per
    unit at layer 23, 7–10 at layer 15 and 2–4 at layers 7–8: early writes are mostly
    cancelled downstream.

### 3. Final-norm gain: a dose-response by task type

The same prompts, the same seed, the gain emulated exactly; differences are paired by prompt.

| Family | n | Gain 0.85 | Gain 1.0 (parent) | Gain 1.2 | 0.85 − 1.0 | gained / lost | 1.2 − 1.0 | gained / lost |
|---|---|---|---|---|---|---|---|---|
| MATH | 120 | 48.3 | 50.0 | 43.3 | −1.7 ± 4.1 | 11 / 13 | −6.7 ± 4.4 | 10 / 18 |
| GSM8K | 80 | 72.5 | 78.8 | 76.2 | −6.2 ± 4.8 | 5 / 10 | −2.5 ± 3.5 | 3 / 5 |
| ARC | 48 | 85.4 | 83.3 | 79.2 | +2.1 ± 4.7 | 3 / 2 | −4.2 ± 5.9 | 3 / 5 |
| CommonsenseQA | 36 | 52.8 | 55.6 | 44.4 | −2.8 ± 10.1 | 6 / 7 | −11.1 ± 8.7 | 3 / 7 |
| MMLU | 36 | 66.7 | 72.2 | 72.2 | −5.6 ± 6.8 | 2 / 4 | 0.0 ± 6.9 | 3 / 3 |
| **Reasoning families** | 320 | **62.5** | **65.3** | **60.3** | **−2.8 ± 2.5** | 27 / 36 | **−5.0 ± 2.4** | 22 / 38 |
| IFEval | 64 | 64.1 | 60.9 | 68.8 | +3.1 ± 5.4 | 7 / 5 | +7.8 ± 5.1 | 8 / 3 |
| Capped completions | 384 | 48 | 38 | 44 | | | | |

- **For reasoning tasks the parent's gain is at the optimum.** A sharper and a flatter sampler
  both lose, and both raise the number of capped completions. The loss is mostly accuracy,
  not budget: among the reasoning prompts that finish at both gains, 74.4% score at gain 1.0
  against 69.2% at 1.2 (273 prompts; 18 gained, 32 lost), and 75.1% against 72.1% at 0.85
  (269 prompts; 22 gained, 30 lost), with nearly unchanged thinking length (median 777 → 809
  and 770 → 829 tokens).
- The three columns share one baseline sample, so part of the dip on both sides can be a
  favourable baseline draw. The conclusion that survives this is the weaker one: no global
  gain step in either direction helps reasoning tasks on this parent.
- A score-function estimate of the slope at gain 1.0 from the baseline samples alone
  (`score_fn.py`) gives −3.6 ± 3.1 points per ×1.05 step: too noisy to add anything.
- IFEval is higher than its baseline at both other gains, which says more about the baseline
  draw (60.9) than about the gain. Its response to a sharper sampler is tested directly, with
  the edited model on fresh prompts, in section 7.

### 4. Outcomes are decided in the thinking

`answer_gain.py` replays each finished IFEval completion up to its `</think>` and re-samples
the answer four times at each of five gains (same random stream across gains, so gains are
compared on common random numbers). 59 of the 60 finished completions were replayed, 236
(completion, draw) cells per gain; answers are cut at 1,000 tokens.

| Answer gain | Score | vs gain 1 | Cells gained / lost | Answers cut at 1,000 tokens |
|---|---|---|---|---|
| 1.00 | 64.4 | | | 9 |
| 1.25 | 66.9 | +2.5 ± 2.2 | 10 / 4 | 12 |
| 1.50 | 65.3 | +0.8 ± 2.1 | 7 / 5 | 17 |
| 2.00 | 65.7 | +1.3 ± 2.1 | 8 / 5 | 17 |
| 3.00 | 64.8 | +0.4 ± 2.5 | 9 / 8 | 25 |

- At gain 1, 53 of the 59 prompts give the same result in all four answer draws (34 always
  pass, 19 always fail). The answer phase is close to a deterministic read-out of what the
  thinking planned: a prompt that fails "no commas" fails it in every draw.
- A gain confined to the answer therefore has a ceiling of two or three points of IFEval,
  reached at ×1.25, and rambling answers become more frequent as the gain rises.
- The same holds for boxed answers: forcing the box right after the close, instead of letting
  the model write its answer out, moves the count of correct answers from 138 to 135 of 185
  (section 2).
- Check of the replay: the row with gain 1 and the stored random stream reproduces the stored
  answer token for token in 58 of 59 cases; the exception is a stored answer of 1,455 tokens,
  identical up to the 1,000-token limit.

### 5. Residual coordinate 0 is softmax-neutral

- The final norm scales coordinate 0 by 0.33; the other coordinates are scaled by 6.3 on
  average. The output embedding barely varies along it (standard deviation across the
  vocabulary 0.016; `</think>`, the end-of-turn token and the other special tokens lie within
  2.5 standard deviations). The direction of least variance of `output embedding × final-norm
  gain` across tokens is 99% coordinate 0: one unit written there spreads the logits by 0.01
  (standard deviation across the vocabulary), against 0.3 for a typical direction
  (`neutral.py`).
- The model uses it: coordinate 0 holds 8.6% of the squared norm of the final residual
  (mean −2.7, standard deviation 1.8, with |h| ≈ 10.3).
- So a write into coordinate 0 changes no logit differences. It only changes the RMS that
  the final norm divides by: a larger coordinate 0 flattens the distribution, a smaller one
  sharpens it. Removing what is already there buys at most ×1.05, so a larger gain has to
  come from the final norm itself, with coordinate 0 used to cancel it where it is not
  wanted.

### 6. A gain that depends on the task type

The edit (`phase_gain.py`), all in the last layer, so it changes output distributions but
never the model's internal state:

- **Final norm:** the gain of every coordinate except 0 is multiplied by G = 1.2. The gain of
  coordinate 0 is lowered (0.328 → 0.133) so that its small logit contribution stays what it
  was on average.
- **Neuron A (always on):** writes `a(x)` into coordinate 0, where `a` is a linear read of
  the MLP input fitted so that `|h + a·e0| = G·|h|`. The RMS normalisation then cancels G.
  A layer-23 neuron cannot see `|h|`, only the normalised input, but a linear read predicts
  the required write well (held-out R² 0.89), which leaves a gain jitter of 1.3–2.5%.
- **Neurons B and C:** the two-neuron SwiGLU step of #66 on a linear feature that separates
  prose-writing tasks from everything else. Where the feature fires they remove what neuron
  A wrote, so the gain G acts. Elsewhere both are exponentially small. The difference of two
  large activations is only formed on the prose side, where precision does not matter.
- The three neurons overwrite the least active neurons of the layer (1040, 1831, 1366:
  largest write over 195k stored states 0.09–0.15, against 1.1 for the median neuron and a
  residual norm of 10). The inherited graft neurons 1998 and 3529 are untouched.

The feature and the reads are fitted on stored states only: thinking and answer states of
the five reasoning families, and of 264 requests written for this purpose
(`make_prose_jobs.py`: 156 prose-writing requests in 11 languages, 48 short-answer, 36
structured-output and 24 code requests). Only the prose requests form the gain class; the
other three kinds are fitted as "keep", so that the gain is confined to tasks that are
clearly about writing prose. The step threshold is placed 0.3 class-gaps toward the prose
side. No IFEval state is used for fitting.

Effective gain on states never used for fitting (a quarter of the completions of each fitted
family, and every IFEval state), from the offline simulation with BF16-rounded weights.
"Gain on" counts states whose effective gain is above √1.2.

| States | n | Gain (mean ± sd) | Gain on | Total variation vs parent |
|---|---|---|---|---|
| MATH, thinking | 13,552 | 1.000 ± 0.013 | 0.0% | 0.002 |
| GSM8K, thinking | 2,710 | 1.002 ± 0.014 | 0.1% | 0.002 |
| ARC, thinking | 1,017 | 1.001 ± 0.021 | 0.6% | 0.004 |
| CommonsenseQA, thinking | 629 | 0.999 ± 0.024 | 1.0% | 0.004 |
| MMLU, thinking | 885 | 1.002 ± 0.023 | 0.9% | 0.003 |
| Code requests, thinking | 442 | 1.002 ± 0.018 | 0.2% | 0.003 |
| Short-answer requests, thinking | 1,165 | 1.002 ± 0.030 | 1.9% | 0.003 |
| Structured-output requests, thinking | 1,145 | 1.014 ± 0.057 | 8.4% | 0.003 |
| Prose requests, thinking | 9,336 | 1.190 ± 0.045 | 94.8% | 0.027 |
| Prose requests, answer | 3,867 | 1.197 ± 0.024 | 98.6% | 0.019 |
| **IFEval, thinking (never fitted)** | 14,213 | 1.156 ± 0.083 | 78.7% | 0.017 |
| **IFEval, answer (never fitted)** | 1,612 | 1.173 ± 0.068 | 86.3% | 0.007 |

- Rounding the neuron activations to BF16 changes none of these numbers in the third
  decimal.
- With the threshold at the midpoint instead, 91% of IFEval thinking states are gated, but
  so are 8% of MMLU and 20% of structured-output thinking states. Since section 3 says a
  sharper sampler costs reasoning tasks, the stricter threshold was chosen.

**The same on the real forward pass** (`verify_gain.py`: six finished completions per family,
teacher-forced through parent and edited model; the gain is the regression slope of the
edited logits on the parent's over the parent's top 64 tokens; total variation is between the
two sampling distributions of the evaluation sampler):

| States | n | Gain (mean ± sd) | 1st–99th percentile | Total variation per token |
|---|---|---|---|---|
| MATH, thinking | 11,741 | 0.999 ± 0.013 | 0.968–1.036 | 0.0017 |
| MATH, answer | 9,493 | 1.000 ± 0.014 | 0.967–1.039 | 0.0015 |
| GSM8K, thinking | 5,985 | 1.000 ± 0.014 | 0.964–1.032 | 0.0021 |
| ARC, thinking | 3,815 | 1.001 ± 0.018 | 0.955–1.042 | 0.0047 |
| CommonsenseQA, thinking | 2,419 | 1.000 ± 0.029 | 0.951–1.170 | 0.0046 |
| MMLU, thinking | 4,417 | 1.003 ± 0.025 | 0.957–1.122 | 0.0037 |
| IFEval, thinking | 8,087 | 1.177 ± 0.064 | 0.964–1.215 | 0.0286 |
| IFEval, answer | 589 | 1.150 ± 0.084 | 0.967–1.213 | 0.0220 |

- The logit pattern is preserved: after removing the gain, edited and parent logits differ
  by 0.02–0.04 (standard deviation over the top 64 tokens).
- Closing and stopping are unchanged on reasoning prompts (`edit_eval.py collect`, then
  `close_stop.py`; 59 finished completions: 28 ARC, 9 CommonsenseQA, 7 GSM8K, 7 MATH, 3 MMLU,
  5 IFEval). At the 107 thinking states of reasoning prompts where the parent's `</think>`
  probability exceeds 0.02, the edited model's is 1.008 ± 0.035 times the parent's. At the
  last state of the 54 reasoning completions, the ratio for the end-of-turn token is
  0.999 ± 0.006. Over all 72,004 reasoning positions in that set the two sampling
  distributions differ by 0.3% per token in thinking and 0.2% in the answer.
- **In BF16 arithmetic** (`verify_bf16.py`: parent and edited BF16 checkpoints run with
  `dtype=bfloat16` on CPU over the first 2,000 generated tokens of two completions per family)
  the picture is the same: thinking-state gain 0.999 ± 0.013 on MATH, 1.000 ± 0.014 on GSM8K,
  1.004 ± 0.019 on ARC, 1.006 ± 0.044 on CommonsenseQA, 1.001 ± 0.021 on MMLU, and a median of
  1.199 on IFEval. CPU kernels are not the platform's GPU kernels, so this checks tolerance
  to 8-bit mantissas, not bit-exact behaviour.
- **Formats the edit was not fitted on** (`make_format_probe.py`, 26 reasoning prompts written
  for this check, 25 of them finished when it was run: few-shot question/answer, step-by-step
  logic puzzles, ten-option multiple choice, true/false, word problems in six other
  languages, plain arithmetic without the boxed instruction). The correct behaviour is "no
  gain", and that is what happens: thinking states have gain 0.999–1.001 (sd 0.015–0.021,
  99th percentile at most 1.05) in every format, with a total variation of 0.001–0.003 per
  token.
- **Tool-calling and code prompts, also not fitted** (`make_tool_probe.py`, 30 prompts
  written for this check: 18 tool-calling prompts rendered with tool schemas the way the
  public BFCL handler renders them, and 12 code prompts in the two wrappers of the public
  HumanEval/MBPP runner). The correct behaviour is again "no gain".

  | States | n | Gain (mean ± sd) | Median | Share above 1.1 | Total variation per token |
  |---|---|---|---|---|---|
  | Code, thinking | 11,755 | 1.001 ± 0.017 | 1.001 | 0.1% | 0.003 |
  | Code, answer | 1,807 | 1.001 ± 0.017 | 1.002 | 0.1% | 0.002 |
  | Tool call expected, thinking | 1,065 | 1.013 ± 0.056 | 0.998 | 8.3% | 0.003 |
  | Tool call expected, `<tool_call>` block | 892 | 0.991 ± 0.063 | 0.981 | 7.1% | 0.000 |
  | No tool applies, thinking | 485 | 1.027 ± 0.064 | 1.005 | 13.0% | 0.007 |
  | No tool applies, prose answer | 605 | 1.079 ± 0.094 | 1.022 | 39.2% | 0.020 |

  Code prompts behave like the reasoning families. On tool-calling prompts most states keep
  gain 1.0, 8% of thinking states are switched to 1.2, and inside the `<tool_call>` block the
  cancelling write is about 2% too strong (1st percentile 0.90). Neither changes what is
  sampled by much: the two sampling distributions differ by 0.3% per token in thinking, and
  not measurably in the block, where the next token is nearly certain. When no listed tool
  applies the model answers in prose, and that answer is partly switched, as designed.

### 7. End-to-end check of the edited model

`fb984a19` and the edited model of section 6 sampled end to end, one sample per prompt, same
seed, paired by prompt. The 136 "fresh" prompts are the IFEval prompts of the pool that no
earlier run or fit had used; the other 64 are those of the job file, whose parent sample is
the baseline of sections 1 and 3.

| IFEval prompts | n | Parent | Edited | Edited − parent | gained / lost | Capped, parent → edited |
|---|---|---|---|---|---|---|
| Fresh | 136 | 72.1 | 72.1 | 0.0 ± 3.5 | 11 / 11 | 9 → 11 |
| Job file | 64 | 60.9 | 65.6 | +4.7 ± 6.1 | 9 / 6 | 4 → 8 |
| Both | 200 | 68.5 | 70.0 | **+1.5 ± 3.0** | 20 / 17 | 13 → 19 |

- **A sharper sampler on prose-writing tasks is not shown to help.** On fresh prompts the
  edited model is level with the parent. The higher score on the 64 job-file prompts repeats
  what the 0.85 and 1.2 arms showed there (section 3): the baseline draw on those prompts was
  low, and any second sample scores higher.
- The edited model caps slightly more often (19 against 13 of 200; 14 prompts capped only
  when edited, 8 only for the parent), which is the direction section 3 found for the
  reasoning families and too few cases to establish.
- A likely mechanism, from a small sample: in the 5 IFEval completions of the closing check
  of section 6, at the 21 thinking states where the parent gives `</think>` a probability
  above 0.02, the edited model gives it 0.65 ± 0.50 times as much. A sharper sampler makes a
  close that is not yet the most likely token less likely still, so thinking runs longer.
- So the gain-1.2 arm of section 3 was misleading for IFEval, and this entry is, as far as
  local evidence goes, neutral: unchanged on reasoning tasks by construction, and within
  noise on the tasks it does change.

### 8. The leaderboard top is a noise plateau

Board as of 2026-11-16 04:40 UTC (`delta leaderboard --model qwen35-08b-v1`).

- `fb984a19` (53.4) has 18 descendants: 15 children and, through `eff1e1b7`, three
  grandchildren. The 19 scores range from 48.6 to 54.3, with mean 52.3 and standard deviation
  1.42. The standard error of one 1,000-item evaluation at that level is 1.58.
- The best of 19 evaluations of one policy with that spread is expected at 54.9. The observed
  best is 54.3, reached by `eff1e1b7` (#80, a 25% rollback of the two graft neurons; its
  record states that no local improvement was demonstrated) and by its child `a6eb9f13`. The
  top of the board is what resampling alone produces.
- Two children are close to re-evaluations of the parent itself. `4b8a20a6` and `1b23b576`
  (#74, #69: eight GRPO steps each) differ from `fb984a19` in 4.5% of the weights, by one
  BF16 step each, with a relative weight change of 3·10⁻⁵. They scored 52.2 and 51.5.
- `856bfe07` (#81) halves the truncation rate, from 14.2% to 7.0%, and scores 50.9. If capped
  items were recoverable at the model's usual accuracy, that step would be worth about four
  points. This is an on-test observation in line with section 2.
- The leaderboard statistics split each score into two halves of 500 items
  (`private_suite_passed` and the rest). Over the 19 entries the halves average 54.0 and 50.5,
  with standard deviations 1.65 and 1.78 and a correlation of +0.36.
- Differences of one or two points between top entries, including the candidate below, are
  not resolved by one evaluation. For a policy whose level is between 52.3 (the 19 entries)
  and 53.2 (the three children of `eff1e1b7`), one evaluation exceeds 54.3 with a probability
  of 7% to 22%.

### 9. The edit on the current leader

While this record was being written `eff1e1b7` (#80) took the top of the board, and the edit
was moved onto it.

**What differs in the new parent.** `eff1e1b7` differs from `fb984a19` in 6,133 values: the
`gate_proj` and `up_proj` rows and the `down_proj` columns of the graft neurons 1998 and
3529. The final norm and neurons 1040, 1831 and 1366 are identical in both. The graft neurons
still matter for this edit, because their output columns point mostly along coordinate 0:
±0.106 of a column norm of 0.132 in `fb984a19`, ±0.08 of 0.12–0.13 after the rollback. They
are active while the model thinks. At the thinking states of the reasoning prompts the
rollback moves coordinate 0 by +0.15 on average (from −2.80) and lowers the norm of the final
residual by 0.23%. Answer states do not move (−0.004).

Measured with the tools of section 6 (`neurons_as_edit.py` expresses the two neurons as an
edit of `fb984a19`), the rollback leaves answer states unchanged and changes thinking
states: the sampling distributions differ by 1.2–2.4% per token, the gain on reasoning
prompts is 1.003–1.005, and at the 52 thinking states where `fb984a19` gives `</think>` a
probability above 0.02 (30 ARC and CommonsenseQA completions) `eff1e1b7` gives it
0.71 ± 0.46 times as much.

**The values fitted on `fb984a19` do not carry over exactly.** Applied unchanged to
`eff1e1b7` (`development_edit.json` builds on either parent), they leave reasoning prompts
with a small gain while the model thinks: 1.002 on MATH, 1.006 on GSM8K, 1.007 on MMLU,
1.008 on ARC and 1.009 on CommonsenseQA, against 0.999–1.003 on `fb984a19`. Answer states
stay at 0.998–1.003. The write that cancels the norm step had been fitted to the old
coordinate 0.

**The fit was therefore repeated on `eff1e1b7`**: same script and arguments, same 373 + 264
completions, teacher-forced through the new parent. The MLP inputs are bit-identical, so the
same three neurons are selected and their gate rows and output columns come out unchanged.
The `up_proj` rows differ from the earlier fit by 18% of their norm, and coordinate 0 of the
final norm goes to 0.1289 instead of 0.1328. Held-out R² of the read is 0.900. On the real
forward pass (`verify_gain.py`, the same 36 completions as in section 6):

| States | n | Gain (mean ± sd) | 1st–99th percentile | Total variation per token |
|---|---|---|---|---|
| MATH, thinking | 11,741 | 0.999 ± 0.012 | 0.970–1.033 | 0.0016 |
| MATH, answer | 9,493 | 0.999 ± 0.013 | 0.968–1.037 | 0.0014 |
| GSM8K, thinking | 5,985 | 1.001 ± 0.013 | 0.968–1.032 | 0.0021 |
| ARC, thinking | 3,815 | 1.001 ± 0.017 | 0.959–1.036 | 0.0046 |
| CommonsenseQA, thinking | 2,419 | 1.001 ± 0.028 | 0.958–1.175 | 0.0047 |
| MMLU, thinking | 4,417 | 1.002 ± 0.025 | 0.959–1.119 | 0.0035 |
| IFEval, thinking | 8,087 | 1.177 ± 0.064 | 0.965–1.215 | 0.0289 |
| IFEval, answer | 589 | 1.150 ± 0.084 | 0.964–1.213 | 0.0223 |

- Closing and stopping (`close_stop.py`, the same 59 completions): at the 90 thinking states
  of reasoning prompts where the parent's `</think>` probability exceeds 0.02, the edited
  model's is 1.006 ± 0.030 times the parent's. At the last state of the 54 reasoning
  completions the end-of-turn ratio is 1.000 ± 0.009. The sampling distributions differ by
  0.30% per token in thinking and 0.21% in the answer.
- Formats not fitted (the probe completions of section 6): 0.998–1.001 while thinking in the
  six reasoning formats; 1.001 ± 0.016 on code; on tool calling 1.012 ± 0.056 while thinking,
  with 8.4% of states switched, and 0.987 ± 0.064 inside the `<tool_call>` block.
- In BF16 arithmetic (`verify_bf16.py`, as in section 6): thinking-state gain 0.999 ± 0.012
  on MATH, 1.000 ± 0.013 on GSM8K, 1.003 ± 0.017 on ARC, 1.006 ± 0.044 on CommonsenseQA,
  1.000 ± 0.020 on MMLU, and a median of 1.199 on IFEval.
- End-to-end sampling (`make_e2e_jobs.py`: 64 reasoning prompts that no run or fit had
  used and 32 of the fresh IFEval prompts of section 7; `eff1e1b7` and the edited model, one
  sample per prompt, same seed): parent 65.6, edited 63.5, a difference of −2.1 ± 4.2
  (7 gained, 9 lost), with 13 against 15 capped completions. The 64 reasoning prompts give
  39 correct against 38 (4 gained, 5 lost), the 32 IFEval prompts 24 against 23 (3 gained,
  4 lost). This check can only exclude a large loss, and it does. The record was opened
  when 85 of the 96 prompts had finished in both arms (−1.2 ± 4.6 at that point).
- The edit does not drift on its own output. Along the 81 finished completions that the
  edited model sampled in that check (`verify_gain.py`), reasoning states keep gain
  1.000 ± 0.020 while thinking (0.7% of states above 1.1) and 0.999 ± 0.017 in the answer;
  IFEval states are switched in 72% of thinking and 83% of answer states.

### Scorer observations (not used to gain score)

- `score_answer` parses the gold answer of a `math` item with the same extraction as a
  model answer. A bare LaTeX gold has no `$…$` or `\boxed{}` anchor: `8x^9`, `(6,7)`,
  `\tan x`, `p - q` parse to nothing, so the item can never score, and `2\sqrt{2}` parses to
  `2`, `9\pi` to `9`, `\left(3, \frac{\pi}{2}\right)` to `3`, so a correct answer scores 0.
  `benchmarks/heldout-v1/build_heldout.py` wraps both sides as `$…$`; `score_answer` does
  not. On the 120 local MATH prompts, bare golds score 40.8% against 50.0% with anchored
  golds, for identical outputs. Whether private golds are anchored is not knowable from
  here; if they are bare, every entry loses the same items.
- The IFEval language checks call `langdetect`, which is randomised unless seeded; the
  same output flipped between two scoring runs of one item.

## Candidate

**Edit.** Parent `eff1e1b7-18ef-4c77-83ef-25fdcc027e75` with four tensors changed, 10,236
values in all: the final norm (1,024 values) and the `gate_proj` rows, `up_proj` rows and
`down_proj` columns of neurons 1040, 1831 and 1366 of the layer-23 MLP (the `down_proj`
columns are zero except in coordinate 0). Gain 1.2 on prose-writing tasks, step threshold
0.3 class-gaps toward the prose side, step steepness 12. This is the fit of section 6
repeated on the states of `eff1e1b7` (section 9).

| Tensor (`model.language_model.` …) | Changed | SHA-256 of the edited BF16 tensor |
|---|---|---|
| `norm.weight` | 1,024 | `9e6b8cd88f104e5c612ffcb1b8d384ec8e7c153ae45b1320f5297106e8481654` |
| `layers.23.mlp.gate_proj.weight` | 3,071 | `f681d9295395f24eb281be28a69629e984515b26095fdc561b881ca62a7988c8` |
| `layers.23.mlp.up_proj.weight` | 3,069 | `0f96ed2d3ebe95954e5b75689e4fd949ebc547bdf8c6333a8ae116df16ed7585` |
| `layers.23.mlp.down_proj.weight` | 3,072 | `9e3e88cf29eead53a3d2dacf41da80b774d307934967d5a26d9700806f915d61` |

- Edited checkpoint (`model.safetensors`) SHA-256
  `dd51c2d6ed6b19112b92117fb554f1473af63465c3a3cdaacadd9690e6404d7d`; the other 484 tensors are
  bit-identical to the parent.
- Patch built with `delta patch build` (format `sparse-diff`, parent content hash
  `6ba2eed0…`): `weights.safetensors` 102,968 bytes, SHA-256
  `8672403bbf194a43e6ed66975ae6d73b91ef71cdce12a4af018c605c1b9168f6`. `delta validate` accepts
  it locally.
- Two text files in this directory hold the edits exactly (see Reproduction).
  `candidate_edit.json` is this candidate. `development_edit.json` is the fit on `fb984a19`
  that sections 6 and 7 measured (10,237 values; checkpoint SHA-256 `be6f2dae…57f1619d` on
  `fb984a19`).

**What is expected, and what is not.**

- On reasoning tasks nothing should change: the sampling distribution of the edited model
  differs from the parent's by 0.2–0.5% per token there (sections 6 and 9), which is far below
  anything section 3 shows to matter.
- On prose-writing tasks the model samples 1.2 times sharper while it thinks and answers.
  Locally that is worth +1.5 ± 3.0 points of IFEval over 200 prompts, and 0.0 ± 3.5 on the
  136 of them that had not been used before (section 7): no established effect.
- **This entry is therefore submitted as a neutral edit, not as a claimed improvement.** Its
  expected test score is its parent's level. One evaluation of `eff1e1b7` gave 54.3, the
  three entries built on it gave 54.3, 53.1 and 52.3, and the 19 entries of this family
  average 52.3 (section 8). The leaderboard number will reflect the resampling noise of one
  evaluation (standard error 1.6). What the entry contributes is the mechanism: a verified
  way to set the sampling gain per state with 10,236 last-layer values, for anyone who finds
  a state in which a different gain pays.

## Reproduction

```sh
# environment: this repository's .[local] extra, Python 3.12, CPU only
export OMP_NUM_THREADS=1 PYTHON=.venv/bin/python DELTA_LAB_MODEL=WORKSPACE/model
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output WORKSPACE
delta competition init qwen35-08b-v1 --parent eff1e1b7-18ef-4c77-83ef-25fdcc027e75 --output TOP

# 1. The exact edits, from their text form (no sampling involved)
$PYTHON edit_json.py apply TOP/model/model.safetensors candidate_edit.json edited.safetensors
(cd TOP && delta patch build --parent model --modified ../edited.safetensors --output ../submission \
   && delta validate ../submission)
$PYTHON edit_json.py records candidate_edit.json CANDIDATE.pt     # local files for --edit below
$PYTHON edit_json.py records development_edit.json DEV.pt         # the fit on fb984a19 (sections 6-7)
$PYTHON edit_json.py apply WORKSPACE/model/model.safetensors development_edit.json edited-fb984a19.safetensors

# 2. Baseline, exit oracles and gain arms (sections 1-3)
$PYTHON to_fp32.py WORKSPACE/model FP32                    # float32 copy shared by all workers
$PYTHON test_engine.py FP32                                # engine versus stock transformers
$PYTHON data.py DATA && $PYTHON make_jobs.py DATA/pool.jsonl JOBS.jsonl
./launch.sh FP32 JOBS.jsonl RUN/leader 8 --samples 1 --seed 1000 --batch 16 --kv-gb 1.0
$PYTHON analyze.py JOBS.jsonl RUN/leader
$PYTHON oracle.py --model FP32 --jobs JOBS.jsonl --base-run RUN/leader --out RUN/exit7800 \
        --plan "reclose@7800,boxafter@7800"
$PYTHON oracle.py --model FP32 --jobs JOBS.jsonl --base-run RUN/leader --out RUN/soft6000 \
        --plan "push@6000:248069:20,boxafter@6000"
$PYTHON oracle.py --model FP32 --jobs JOBS.jsonl --base-run RUN/leader --out RUN/box3000 \
        --plan "boxafter@3000" --late-close 3000           # also 0, 4500, 6000
$PYTHON oracle_report.py JOBS.jsonl RUN/leader RUN/exit7800 RUN/soft6000 RUN/box3000
./launch.sh FP32 JOBS.jsonl RUN/gain120 8 --samples 1 --seed 1000 --batch 16 --kv-gb 1.0 --gain 1.2
./launch.sh FP32 JOBS.jsonl RUN/gain085 8 --samples 1 --seed 1000 --batch 16 --kv-gb 1.0 --gain 0.85
$PYTHON compare_runs.py JOBS.jsonl RUN/leader RUN/gain120
$PYTHON compare_runs.py JOBS.jsonl RUN/leader RUN/gain085

# 3. Answer replay (section 4)
$PYTHON answer_gain.py --model FP32 --jobs JOBS.jsonl --base-run RUN/leader --out RUN/ansgain \
        --families ifeval --gains 1,1.25,1.5,2,3 --draws 4
$PYTHON answer_gain_report.py JOBS.jsonl RUN/leader RUN/ansgain --ifeval-detail

# 4. The edit (sections 5-6): states, fit, checkpoint, verification
$PYTHON make_prose_jobs.py PROSE.jsonl
./launch.sh FP32 PROSE.jsonl RUN/prose 8 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8
$PYTHON acts.py --model FP32 --jobs JOBS.jsonl --run RUN/leader --out ACTS/design \
        --sites mlp_in:6,mlp_in:7,mlp_in:23,resid:23 --every 8 --after-close 3
$PYTHON acts.py --model FP32 --jobs PROSE.jsonl --run RUN/prose --out ACTS/prose \
        --sites mlp_in:23,resid:23 --every 4 --after-close 1500 --only ok
$PYTHON neutral.py FP32 ACTS/design
$PYTHON phase_gain.py --model FP32 --acts ACTS/design ACTS/prose --gain 1.2 --shift=-0.3 \
        --gain-class prose:all \
        --keep-class math:all,gsm8k:all,arc:all,csqa:all,mmlu:all,short:all,struct:all,code:all \
        --measure-only ifeval --save FIT.pt                # the fit; DEV.pt holds the same records
$PYTHON build_candidate.py WORKSPACE/model/model.safetensors FIT.pt refit-fb984a19.safetensors
$PYTHON verify_gain.py --model FP32 --edit DEV.pt --jobs JOBS.jsonl --run RUN/leader --per-family 6
$PYTHON edit_eval.py collect --model FP32 --edit DEV.pt --jobs JOBS.jsonl --run RUN/leader \
        --out RUN/edit-eval --only ok --max-rows 60 --shard 0/2     # 30 completions
$PYTHON edit_eval.py collect --model FP32 --edit DEV.pt --jobs JOBS.jsonl --run RUN/leader \
        --out RUN/edit-eval2 --only ok --shard 5/12                 # 29 more, spread over families
$PYTHON close_stop.py RUN/edit-eval RUN/edit-eval2
$PYTHON verify_bf16.py WORKSPACE/model edited-fb984a19.safetensors JOBS.jsonl RUN/leader --per-family 2 --tokens 2000
$PYTHON make_format_probe.py PROBE.jsonl
./launch.sh FP32 PROBE.jsonl RUN/probe 2 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8
$PYTHON verify_gain.py --model FP32 --edit DEV.pt --jobs PROBE.jsonl --run RUN/probe --per-family 6
POOL="code=code_complete,code_task;tool call expected=tool_single,tool_multiple,tool_parallel,tool_followup;no tool applies=tool_irrelevant"
$PYTHON make_tool_probe.py TOOLS.jsonl
./launch.sh FP32 TOOLS.jsonl RUN/tools 4 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8
$PYTHON verify_gain.py --model FP32 --edit DEV.pt --jobs TOOLS.jsonl --run RUN/tools --per-family 6 --pool "$POOL"

# 5. End-to-end check (section 7): IFEval.jsonl holds the IFEval prompts of DATA/pool.jsonl
./launch.sh FP32 IFEval.jsonl RUN/ifeval-parent 8 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8
./launch.sh FP32 IFEval.jsonl RUN/ifeval-edited 8 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8 --edit DEV.pt
$PYTHON compare_runs.py IFEval.jsonl RUN/ifeval-parent RUN/ifeval-edited

# 6. The edit on the current leader (section 9): same completions, states of the new parent
$PYTHON to_fp32.py TOP/model FP32TOP
$PYTHON acts.py --model FP32TOP --jobs JOBS.jsonl --run RUN/leader --out ACTS/design-top \
        --sites mlp_in:23,resid:23 --every 8 --after-close 3
$PYTHON acts.py --model FP32TOP --jobs PROSE.jsonl --run RUN/prose --out ACTS/prose-top \
        --sites mlp_in:23,resid:23 --every 4 --after-close 1500 --only ok
$PYTHON phase_gain.py --model FP32TOP --acts ACTS/design-top ACTS/prose-top --gain 1.2 --shift=-0.3 \
        --gain-class prose:all \
        --keep-class math:all,gsm8k:all,arc:all,csqa:all,mmlu:all,short:all,struct:all,code:all \
        --measure-only ifeval --save FIT-TOP.pt            # the fit; CANDIDATE.pt holds the same records
$PYTHON verify_gain.py --model FP32TOP --edit CANDIDATE.pt --jobs JOBS.jsonl --run RUN/leader --per-family 6
$PYTHON verify_gain.py --model FP32TOP --edit CANDIDATE.pt --jobs PROBE.jsonl --run RUN/probe --per-family 6
$PYTHON verify_gain.py --model FP32TOP --edit CANDIDATE.pt --jobs TOOLS.jsonl --run RUN/tools --per-family 6 --pool "$POOL"
$PYTHON edit_eval.py collect --model FP32TOP --edit CANDIDATE.pt --jobs JOBS.jsonl --run RUN/leader \
        --out RUN/top-eval --only ok --max-rows 60 --shard 0/2
$PYTHON edit_eval.py collect --model FP32TOP --edit CANDIDATE.pt --jobs JOBS.jsonl --run RUN/leader \
        --out RUN/top-eval2 --only ok --shard 5/12
$PYTHON close_stop.py RUN/top-eval RUN/top-eval2
$PYTHON verify_bf16.py TOP/model edited.safetensors JOBS.jsonl RUN/leader --per-family 2 --tokens 2000
$PYTHON verify_gain.py --model FP32TOP --edit DEV.pt --jobs JOBS.jsonl --run RUN/leader --per-family 6  # earlier values, unchanged
$PYTHON neurons_as_edit.py FP32TOP/model.safetensors 1998,3529 ROLLBACK.pt   # eff1e1b7 as an edit of fb984a19
$PYTHON verify_gain.py --model FP32 --edit ROLLBACK.pt --jobs JOBS.jsonl --run RUN/leader --per-family 6
$PYTHON edit_eval.py collect --model FP32 --edit ROLLBACK.pt --jobs JOBS.jsonl --run RUN/leader \
        --out RUN/rollback-eval --only ok --max-rows 60 --shard 0/2
$PYTHON close_stop.py RUN/rollback-eval --changed none
$PYTHON make_e2e_jobs.py DATA/pool.jsonl JOBS.jsonl E2E.jsonl                 # 96 prompts outside the job file
./launch.sh FP32TOP E2E.jsonl RUN/e2e-parent 4 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8
./launch.sh FP32TOP E2E.jsonl RUN/e2e-edited 4 --samples 1 --seed 1000 --batch 16 --kv-gb 0.8 --edit CANDIDATE.pt
$PYTHON compare_runs.py E2E.jsonl RUN/e2e-parent RUN/e2e-edited
$PYTHON verify_gain.py --model FP32TOP --edit CANDIDATE.pt --jobs E2E.jsonl --run RUN/e2e-edited --per-family 32  # on its own samples
```

- `pool.jsonl` SHA-256 begins `d3914f4aefb66bae`; the 384-prompt job file begins
  `dcbcdba901559f88`. Dataset revisions are whatever the Hub served on 2026-11-13, so the
  hashes are the check.
- `candidate_edit.json` and `development_edit.json` are exact: they hold every changed value
  as a bfloat16 bit pattern and the SHA-256 of the four resulting tensors per parent, and
  `edit_json.py apply` reproduces the checkpoints byte for byte (`dd51c2d6…` for the
  candidate on `eff1e1b7`; `be6f2dae…` for the development edit on `fb984a19` and
  `13030a14…` for the same values on `eff1e1b7`). It refuses a parent whose result matches
  none of the recorded hashes. Steps 4 and 6 regenerate the two edits from stored states;
  the first was reproduced bit for bit on this machine, but sampling is only deterministic
  given the seed, the batch composition and the machine, so on other hardware the fitted
  values can differ in the last bits. Use step 1 for the artifacts and steps 4 and 6 to
  check the method.
- The design states were captured when 373 of the 384 baseline completions had finished; the
  11 missing ones are long capped MATH completions. Step 6 used the same 373 completions.
- The 0.85 arm was run as two jobs (the 320 reasoning prompts on 8 workers, the 64 IFEval
  prompts on 3). Section 7 was run as two prompt sets: the 64 IFEval prompts of the job file
  (edited model on 3 workers) and the 136 other IFEval prompts of the pool (4 workers per
  model).
- A 384-prompt run takes about 3 hours on 8 single-thread workers. In this VM PyTorch is
  10–60 times slower with more than one thread per process.

## Limitations

- Local prompts come from public train splits, which the parent's ancestors may have seen
  during GRPO or distillation (#58 trains on these splits and on IFEval). Absolute scores are
  therefore optimistic; paired differences are what the conclusions rest on.
- One sample per prompt. The gain results are single paired comparisons with standard errors
  of 2 to 6 points. They exclude large effects in either direction, not small ones.
- The private test's task mix and prompt formats are not public. The edit is meant to change
  behaviour only on prose-writing tasks, and it relies on the states of other task types
  falling on the "keep" side of a feature fitted on the formats above. The probes are small:
  26 prompts in six other reasoning formats and 12 code prompts fell on that side; 18
  tool-calling prompts mostly did, with 8% of their thinking states switched (section 6).
- The patch's parent changed late. Sections 1 to 7 are measurements on `fb984a19`. On
  `eff1e1b7` the fit and the forward-pass checks were repeated (section 9), the gain arms
  and the answer replay were not, and end-to-end sampling was limited to 96 prompts.
- The prose-writing feature was fitted on 156 requests written by the same agent that
  designed the experiment. IFEval is the only independent check of where it fires.
- Float32 arithmetic locally, BF16 on the platform. The edit was checked with BF16-rounded
  activations in the simulation and in a BF16 forward pass on CPU, not on the platform's GPU
  kernels.

## Submission

[Delta League submission](https://deltaleague.ai/submissions/34bf31fc-bda8-466d-a6e5-ddd13a967fb8/),
ID `34bf31fc-bda8-466d-a6e5-ddd13a967fb8`, submitted on 2026-11-16 at 07:04 UTC with the URL of
this PR (#83).

**Official result** (2026-11-16 07:30 UTC, version `bde3d8e4-77a8-4605-bbd7-b96f7a2db4ac`):
test score **53.5** with standard error 1.58, rank 3 of 77 on that day. Truncation rate
15.0%, public validation 46.5.

| | This entry | Parent `eff1e1b7` |
|---|---|---|
| Test score | 53.5 | 54.3 |
| Passes in the two 500-item halves | 275 + 260 | 285 + 258 |
| Truncation rate | 15.0% | 14.2% |
| Public validation, 18 benchmarks | 46.5 | 47.1 |
| Public validation, IFEval (64 items) | 46 | 50 |

- The entry is 0.8 points below its parent's one evaluation, half the standard error of a
  single evaluation. It is neutral as far as one evaluation can tell, which is what the
  record predicted, and it is not an improvement.
- The four entries built on `eff1e1b7` now score 54.3, 53.5, 53.1 and 52.3.
- Public IFEval gives no sign of a gain from the sharper sampler either: 46 of 64 against
  50 for the parent, inside the range of 44 to 52 that the entries of this family span.

The platform cut the submission title at 160 characters. Its last words were meant to read
"local IFEval +1.5 +/- 3.0, not established".

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs.
- [x] Local estimates and verified results are clearly distinguished.
