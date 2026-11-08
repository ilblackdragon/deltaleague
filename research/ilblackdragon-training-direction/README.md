# Training-data teacher-guided direct weight edit

Status: interrupted unexpectedly during the student training-data baseline; incomplete results, no candidate or submission.

## Hypothesis

Illia Polosukhin proposed learning from larger-model successes to improve a small
model, then requested separate training-like data rather than repeatedly adapting
to League validation. Contrast failed student reasoning with successful teacher
reasoning on several training problems, replay both in the original student, and
partially remove their average difference direction from two student MLP output
matrices. This is a hypothesis: correctness contrasts may encode task, style,
position or completion differences rather than a reusable failure mechanism.

This experiment supersedes neither prior result nor its protocol. The earlier
1,536-token diagnostic is in PR #15 and is not evidence about this 8,192-token run.

## Frozen data and protocol

Use only `openai/gsm8k`, `main`, **train** split at revision
`740312add88f781978c0658806c59bc2815b9866`. Shuffle indices with Python seed
20261106, skip duplicate normalized questions, select 16 training problems and
eight separate held-out problems. IDs, document hashes and partitions are in
`selection.json`. Preparation fixes the selection file hash before inference.
No League validation examples or outcomes influence this experiment. Dataset
loading may materialize other public splits in the cache; only train rows are
selected or used. No hidden data is accessed.

No demonstrations are used. The private worker's actual few-shot choices are
unavailable, so we cannot claim exclusion from its unknown demonstration draws.
This is an independent zero-shot experiment, not an exact League reproduction.

All models receive the identical question and request for step-by-step reasoning
and a boxed answer, with their own chat templates and thinking enabled. Generation
seed 1000, temperature 1, top-p .95, top-k 20, generated-token presence penalty 1.5,
and **8,192 maximum output tokens for every model/example**. There is no token-cap
reduction to finish within the wall-clock budget. CPU Transformers inference uses
float32 for the student and BF16 for the teacher; this precision difference limits
model-size interpretations. The public CLI math scorer checks final answers after
reasoning closes. Missing `</think>` means zero final-answer credit. Repetition is
reported but does not determine pair eligibility in this experiment.

The teacher runs on all 16 training examples, not only student failures. Require
at least three completed, nontruncated, correct-teacher/incorrect-student pairs
after both models complete the entire training set. Final-answer correctness
does not certify every intermediate reasoning step. If fewer than three pairs
exist, stop without constructing an edit. Never select pairs from partial
training coverage.

## One fixed intervention

Replay both reasoning traces in the original student. For layers 15 and 19,
subtract mean down-projection outputs on the teacher reasoning's last 256 tokens
from outputs on the student reasoning's last 256 tokens. Average equally over
eligible training pairs, normalize to `u`, and rank columns of `W` by
`abs(u @ W) * RMS(student tail input activations)`. Select 64 columns per layer:

```text
delta = -0.5 * outer(u, u @ W_selected)
scale = min(1, 0.005 * ||W||_F / ||delta||_F)
W_selected_new = BF16(W_selected_float32 + scale * delta)
```

Only these two MLP matrices may change. No optimizer, gradients, LoRA, cross-size
weight subtraction, parameter search or holdout-driven retuning. Layer choice and
strength are inherited hypotheses, not experimentally established optima. BF16
rounding can slightly alter the realized norm and rank.

The edit routine reads only training generation files. After saving the single
candidate, build and validate its patch and verify bitwise reconstruction. Then
evaluate original student, edited student and teacher on the eight held-out
examples. Nothing learned from those outcomes changes the candidate. Eight
examples provide limited evidence and cannot establish broad improvements.

## Identity and reproduction

- Parent: `qwen35-08b-v1`, version `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`.
- Parent hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Evaluation identity retained: `public-ladder500-8k-v2`; no official scoring claim.
- Teacher: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Original checkpoint and metadata preserved. `edited.safetensors` is a separate file.

Python 3.12, Linux, Ryzen 9 9950X3D; dependencies in `requirements.lock.txt`.
Install CPU PyTorch from its CPU wheel index before the dependency lock. Initialize
the pinned original parent with the CLI and reuse the pinned teacher checkpoint.
Copy `plan.json` to the workspace; fresh runs need their own explicit budget and
start/deadline timestamps. Do not reuse these historical timestamps or restart
the budget automatically.

```bash
delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output WORKSPACE
mkdir -p WORKSPACE/local
# Place the pinned teacher directory at WORKSPACE/teacher.
python research/ilblackdragon-training-direction/prepare.py --workspace WORKSPACE
python research/ilblackdragon-training-direction/pipeline.py --workspace WORKSPACE
```

The pipeline runs stages sequentially in separate processes, avoiding accumulated
model memory. The approved ceiling is **12 hours, 16 compute threads, 64 GiB
address space per model process, no paid compute**. Process alarms, generation
checks and subprocess timeouts enforce one shared absolute deadline. It stops on
error; it never rents compute or submits. Completed generation rows are flushed
after each problem, and timeout/partial coverage is explicitly recorded. No
automatic new budget or parameter retries. Existing weights are reused via local
symlinks; no symlinks, weights, raw traces or credentials are committed here.

`report.py` exports coverage and summaries without raw generation text.
`audit.py` verifies parent artifacts and (when present) candidate scope,
finiteness, and patch reconstruction. All seven parent artifacts passed before
launch. Source hashes and selection were frozen before inference.

## Findings and submission

The process stopped unexpectedly with last logged progress at 2026-11-06 01:57:57
UTC. On inspection at 23:43 UTC, neither the pipeline nor its inference child was
alive and the execution session no longer existed. No final exception or shutdown
record explains the termination. The old running marker was stale; it has been
corrected. This was not a verified normal timeout. The approved deadline was
12:11 UTC and has now passed. No restart or new budget was assumed.

Six of 16 student training-data examples completed: one correct final answer
(5,266 generated tokens), five capped at 8,192 tokens without closing reasoning.
The seventh trace stopped during generation, with 5,888 tokens last logged; it
is excluded from scored results. The completed subset's 1/6 accuracy is partial
coverage, not an estimate from a completed experiment. These data show that long
unfinished reasoning persisted on five examples at 8K; they do not establish
teacher benefit or a generalization result.

The teacher was never run: 0/16 queried, accuracy unknown. No edit was constructed,
no held-out examples evaluated, no patch built and no submission made. No official
validation or private score exists. `results.json` records completed denominators;
`interruption.json` records the observed interruption without inventing a cause.
The initial parent artifact audit passed. Original weights remain untouched by
this experiment.

A continuation requires a newly agreed local budget and a durable execution
mechanism; the expired job must not be represented as still running. Submission
would still require separate explicit quota approval after a concrete candidate.
