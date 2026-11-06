# Teacher-guided direct removal of a repetitive-reasoning direction

**Completed: one direct edit built and validated; no local correctness gain.**
The initial short-budget diagnostic stopped without an edit. In the explicitly
authorized continuation, the teacher completed correctly at 2,882 tokens,
providing the intended contrast for a single gradient-free edit.

The initial stopping outcome is retained as the first protocol's result. After the
user requested continuation, a second diagnostic pass was authorized within the
remaining original two-hour window. It extends only the teacher's output cap to
4,096 tokens. The saved student baseline is reused; the proposed single edit,
student 1,536-token cap, prompt, seed, and other generation settings remain fixed.
`continuation-plan.json` records the explicit amendment. The comparison examples
have already been observed in the first run, so they must not be described as a
new untouched holdout. Candidate evaluation runs the four reserved
examples first and then the four development examples under the same deadline.

For this continuation, use a separate workspace with copies of the original
configuration, selected examples, baseline traces and inspected diagnosis; reuse
the original parent and teacher directories. Preserve the original absolute
deadline in its plan. The teacher command is:

```bash
python research/ilblackdragon-teacher-loop-direction/run.py --workspace CONTINUATION --stage teacher --max-new-tokens 4096
```

The original results and trace summaries are retained rather than overwritten.

The extended teacher's first 1,536 token IDs match the original capped trace
exactly. It then closes reasoning and answers 12 correctly before its new
4,096-token ceiling. `continuation-provenance.json` verifies prefix identity and
bitwise reuse of the student baseline. Its tail repetition coverage is 17.97%,
below the fixed 20% threshold. The full successful teacher response supplied the
contrast; the documented fallback below was not used.

The teacher's larger output allowance is for obtaining a diagnostic contrast;
it is not a matched-budget comparison of model sizes. The original and edited
student use identical 1,536-token budgets and sampling settings. A student given
more tokens might also finish; that question is not answered by this run.

The direct edit changes 61,283 BF16 entries in layer 15 and 61,566 in layer 19
(122,849 total). Realized relative norm changes are 0.50054% and 0.50042% after
rounding. Construction took 9.04 seconds with no optimizer or gradient training.
`direct-edit.json` contains exact column selections and scaling factors.
The 1,228,810-byte sparse patch passed CLI validation, and its reconstruction
matches the full edited checkpoint bit for bit. Original model hashes pass.
`continuation-audit.json` records the checks and artifact hashes.

A segment-based fallback was proposed and timestamped before the extended query
completed, but was unnecessary and never executed. The final reproduction code
retains the successful-full-response requirement; the unused fallback code was
removed. The proposal remains recorded in `continuation-plan.json` for traceability.

## Protocol correction before further inference

The user correctly challenged the 1,536-token diagnostic. It cannot establish
whether this candidate improves the pinned official evaluation. Increasing only
the token cap would still leave prompt and scoring mismatches: the manifest names
native GSM8K with five demonstrations and exact-match metrics, whereas this run
used zero-shot boxed-answer prompts and the CLI math scorer with an added
reasoning-close gate. The earlier results remain valid only for that diagnostic.

Teacher coverage was one queried problem, successfully completed with a longer
allowance, not one correct out of eight. Seven teacher outcomes are unknown.
The next comparison must evaluate original, frozen candidate, and teacher on the
same examples with the same output allowance and verified task setup.

`protocol-parity-audit.json` records the pinned settings, mismatches and remaining
requirements. The public CLI currently rejects this manifest format. Upstream
lm-eval v0.4.13 supplies the native task, but its greedy defaults differ from the
pinned evaluation's sampling configuration; the worker's actual overrides,
few-shot seeds, formatting, stopping and filtering must be verified rather than
inferred. Anonymous access to the pinned worker registry returned 401. No private
worker data or tests were accessed. No new inference has been launched; the old
local-compute budget has expired. A new bounded budget must be agreed once this
public evaluation recipe is available.

## Hypothesis and access

Illia Polosukhin proposed using a larger Qwen model's behavior to improve the
smaller model and requested a direct weight edit inspired by
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).
Codex implements this bounded diagnostic experiment without gradient training.

The hypothesis is that a direction associated with repeated student reasoning
can be identified by contrasting that reasoning with a correct teacher answer,
then partially removed from selected student MLP output columns. Teacher text is
replayed through the student: neither weights nor activations are subtracted
between models of different sizes. This is a candidate causal intervention, not
proof that a contrast direction uniquely represents repetition.

Only public checkpoints, public repository files, and the downloaded public
validation manifest and its public upstream dataset selections are used. No
League private examples are accessed. The upstream GSM8K split called `test` is
public data explicitly named in that public manifest, not the League private test.

Layers 15 and 19 are chosen from the post's prior result; they are not independently
discovered here. This experiment does not reproduce the authors' released diff.

## Parent and evaluation

- Competition: `qwen35-08b-v1`.
- Original parent: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`.
- Parent content hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Pinned evaluation: `public-ladder500-8k-v2`.
- Public validation SHA-256: `996573d687b0d0f1e088f1e918cd153fa0f80cff454a7e9a7ab5ac2eb374d4d2`.
- Teacher: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Public GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`.

The original checkpoint and `delta-model.json` remain unchanged. The local
workspace reuses the already downloaded model and teacher via directory symlinks;
no symlinks or model artifacts are included in this research PR.

## Fixed design

Shuffle the 64 public GSM8K manifest positions with Python `random.Random(42)`;
use the first four for development and next four for holdout. Verify every
selected document's SHA-256 over sorted compact UTF-8 JSON before inference.
Freeze the split before model execution. No holdout trace influences selection
of teacher queries, directions, columns, or edit strength.

Each prompt requests step-by-step reasoning and a boxed final answer. Use each
model's chat template with thinking enabled, no few-shot examples, seed 1000,
temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, and at most 1,536 new
tokens. Presence penalty applies only to generated tokens. Student baseline and
candidate run in float32; the teacher runs in BF16 on the local AVX512-BF16 CPU.

This is a local diagnostic proxy, not the official five-shot GSM8K metric or
8,192-token evaluation. Correctness uses the CLI's public math scorer only after
the generated reasoning has a closing `</think>` tag. A truncated reasoning trace
without a final answer is scored zero, even if a correct number appears inside
the reasoning. Completion, truncation, repetition, and correctness are reported
separately; a token-limit hit alone is not called a loop.

The prespecified repetition detector finds generated 8-token sequences occurring
at least three times. It marks their token positions, then flags a loop when at
least 20% of the final 256 generated tokens are marked. Detected development
traces must also be inspected to distinguish genuine repetition from legitimate
repeated arithmetic. Query the teacher on the first two incorrect flagged
development examples, at most. Only a correct, nonrepetitive teacher answer
paired with an inspected repetitive incorrect student answer supports an edit.
If no such pair is available, stop and report no supported edit.

For supported pairs, replay the student's trace and the teacher's reasoning
(before `</think>`) through the original student. At each target layer, contrast
the mean down-projection output on the student's flagged tail tokens with the
mean output on the teacher reasoning's final 256 tokens. Average contrasts
equally across paired examples and normalize to direction `u`. Rank columns by
`abs(u @ W) * RMS(loop input activation)` and select 64 columns. Apply:

```text
delta_selected = -0.5 * outer(u, u @ W_selected)
scale = min(1, 0.005 * ||W||_F / ||delta_selected||_F)
W_selected_new = BF16(W_selected_float32 + scale * delta_selected)
```

Before replay, remove the teacher's generated opening `<think>` marker because
the student's generation template already includes it. This formatting fix was
made before any edit construction during the continuation; neither model's
generation or the earlier diagnostic results changed.

Only the two down-projection matrices may change. There is one fixed candidate,
no gradient updates, no optimizer, no strength search, and no retuning from
holdout results. BF16 rounding can slightly change the realized norm and rank.
The teacher/student contrast can capture topic, position, or style differences
as well as repetition; paired examples reduce but do not eliminate that confound.

## Reproduction and budget

Dependencies are in `requirements.lock.txt`. Python 3.12, Linux, Ryzen 9 9950X3D,
at most 16 CPU compute threads, 64 GiB address-space limit per inference process.
Stages run sequentially. The approved budget is two hours and 5 GiB additional
disk, reusing cached teacher weights; no rented compute or automatic submission.
`plan.json` records the fixed design and approval. A shared absolute deadline is
enforced by a process alarm and generation-time checks. Raw traces stay under
the local workspace's `local/` directory and must not be committed.

Initialize a new workspace with the exact parent and ensure its evaluation/hash
match the pinned identifiers above. Reuse the pinned teacher snapshot, or download
it separately under an appropriate download/storage budget. Set `plan.json`'s
`started_at_unix` and `deadline_unix` to the authorized start and start plus 7200.
Use `continuation-plan.json` as the plan template to reproduce the actual edit;
the old timestamps are provenance, not a reusable runtime window. Install the
pinned CPU PyTorch wheel from `https://download.pytorch.org/whl/cpu` before the
dependency lock file. Initialize with:

```bash
delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output WORKSPACE
```

```bash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage baseline
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage teacher --max-new-tokens 4096
# Inspect flagged development traces. If there are valid pairs, write
# local/diagnosis.json with confirmed_loop_ids before constructing an edit.
# For the identical seeded reproduction, continuation-diagnosis.json records
# the inspected example. Omitting --max-new-tokens reproduces the initial stop.
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage edit
# Only if an edit was supported and edited.safetensors exists:
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage candidate
cd WORKSPACE
delta patch build --parent ./model --modified ./edited.safetensors --output ./submission
delta validate ./submission
```

`report.py --workspace WORKSPACE` exports aggregate results and IDs without text
or token arrays. `audit.py --workspace WORKSPACE` checks parent file hashes; if
a candidate exists it also checks names/shapes/dtypes, finiteness, target scope,
and bitwise equality of patch reconstruction with `edited.safetensors`.

## Initial 1,536-token diagnostic, preserved

The first run produced:

| Local diagnostic group | Problems | Correct final answers | Hit 1,536-token cap | Strict repetition flags |
| --- | ---: | ---: | ---: | ---: |
| Original development | 4 | 0 | 4 | 1 |
| Original holdout | 4 | 0 | 4 | 0 |
| Teacher on eligible development problem | 1 | 0 | 1 | 0 |

There were no candidate scores at that stage; they are null, not zero. The exact public question
IDs, document hashes, settings, aggregates and per-example metric summaries are
in `results.json`. Raw traces are kept outside Git.

On public problem 421, both models computed 12 cats, then repeatedly reconsidered
whether chasing a mouse meant leaving the rock. Neither closed its reasoning or
gave a final answer before the cap. The student's exact-tail repetition coverage
was 29.30%; the teacher's was 10.16%. A low exact-repetition score therefore did
not guarantee progress. Development problems 438 and 170 also showed repeated
reconsideration with varied phrasing but did not qualify under the fixed detector.
These observations are recorded in `diagnosis.json`; the threshold was not tuned.

The larger teacher was not a successful contrast under that short allowance.
The direct-edit routine was not exercised in the initial diagnostic. The
authorized continuation above subsequently obtained a successful full response
and exercised the edit. The initial stop does not establish that either model
would fail under a longer output allowance.

Baseline generation took 1,708.41 seconds; teacher generation took 501.33 seconds.
Including loading, the baseline and teacher stages took 1,709.83 and 502.31
seconds. End-to-end wall time including preparation and the requested source
review was about 41 minutes, within the two-hour ceiling. Peak measured RSS was
14.95 GiB for the BF16 teacher. All seven original downloaded model files passed
their size/hash checks. At the initial stopping point no weight had changed and
no patch or submission existed. The user later requested continuation within the
remaining original budget; that protocol amendment and its candidate are recorded
separately above. No evaluation quota has been consumed.

## Completed direct-edit comparison

All eight matched comparisons completed under the original two-hour deadline.
These are local diagnostic results at 1,536 tokens, not official League scores.
The reserved set had already been observed during the initial diagnostic; no
candidate parameters were selected or changed using its results.

| Group | Correct original → edited | Completed original → edited | Mean exact tail repetition original → edited |
| --- | ---: | ---: | ---: |
| Development (4) | 0/4 → 0/4 | 0/4 → 0/4 | 11.43% → 4.59% |
| Reserved comparison (4) | 0/4 → 0/4 | 0/4 → 0/4 | 6.74% → 10.55% |

Every student trace reached its cap. There were zero paired accuracy wins or
losses, and four ties in each group. The exact paired sign test is uninformative
(p=1 with zero discordant pairs); this is not evidence of equivalence. Strict
repetition flags fell from one to zero on development and stayed zero on the
reserved set. Mean reserved-set repetition increased. There is no demonstrated
completion, accuracy, or general repetition improvement.

`continuation-results.json` includes complete per-example metrics, token-prefix
divergence, paired comparisons, artifact metadata and runtime summaries. The
baseline was reused bit for bit, not rerun. The extended teacher stage took
987.99 seconds, construction 9.04 seconds, and candidate generation 1,406.36
seconds. Total model-stage time including the initial baseline and teacher was
about 77 minutes; elapsed wall time including preparation and the gap before
continuation was 112.2 minutes. The largest measured inference RSS was
15.31 GiB; the workspace occupies about 1.7 GiB without following reused model
symlinks. The approved limits were 120 minutes, 64 GiB RAM, and 5 GiB extra disk.

The candidate checkpoint SHA-256 is
`50baa498548c1f21f3751545084f38184770156e7318570c126bbea340cf1970`.
The patch SHA-256 is
`a3f97146e9082b1bc82f557b5c9fd13f2e89b9e68afa71dfec37196a6ede9fb0`.
Weights and raw traces are deliberately excluded from the PR. No submission has
been made and no official evaluation is pending. Quota checked after the local
comparison remains 0/5 used.

## What the intervention teaches us

The layer indices refer only to the small model. Their choice comes from prior
work, not a discovered correspondence to the larger model. Different sizes and
architectures need not share neuron coordinates or layer roles. Teacher text
provides a common interface: replaying it in the student yields comparable
student-space activations without pretending the two weight spaces align.

Direct edits can alter sampled behavior with a small matrix perturbation, but
that alone does not establish an improvement or a uniquely identified function
for a layer. This run jointly changes two layers and has no layer ablation, so it
cannot assign the effect to either layer separately. The direction can mix
repetition, topic, style and token-position information.

On development problem 421, exact tail repetition falls from 29.30% to zero,
but inspection of the edited trace still shows repeated reconsideration of the
same arithmetic and interpretation. It does not close its reasoning within the
cap. Exact token repetition is therefore an incomplete proxy for semantic loops;
lower repetition should not be reported as better reasoning or task accuracy.

## Interpretation and submission

Four public holdout examples cannot establish general capability improvement.
The short token cap can prevent answers that would complete under the official
budget, or miss loops that emerge later. Sampling and CPU/precision differences
also prevent interpreting these local figures as official scores.

No submission is authorized by the local-compute approval. Any resulting patch
must first be documented in this research PR and separately approved for quota
consumption. Candidate official scores remain unavailable until a submitted
evaluation completes. Negative or inconclusive diagnoses are retained.

For context only, the existing original League result on `public-ladder500-8k-v2`
is 41.830071900430084 public / 18.6 private. Those platform-verified scores are
different measurements from this short-budget local diagnostic. There is no new
official score or pending evaluation from this run.

## Additional references requested during execution

The user supplied [Zenodo record 23958070](https://zenodo.org/records/23958070)
and [Boris Knyazev's post](https://x.com/borisaknyazev/status/2118065460377792918)
while the original-model baseline was running. `references.json` records source
identity, retrieval details, hashes, methodological caveats, and possible future
experiments. These sources did not change the fixed design of this run. The
third-party documents, media, and code are not redistributed in this PR.
