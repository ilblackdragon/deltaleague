# Editing the 96b67023 mixture

The parent is udirobert's 96b67023, ranked first when this experiment started. The official public score was 49.59 and the private score was 54.80. The title names a 50/50 mixture of 917d6b4f and a6eb9f13. The platform records 917d6b4f as the direct parent. Both sources are public. Recreating their FP32 midpoint, rounding once to the original tensor dtypes and using the organizer's header produced the published 1,746,942,600-byte file, SHA-256 daea46fd6baf4a7e36581325a3a6bce7f7727cce812033ace2e0b08a05a09b43. The six auxiliary files also matched the official metadata. Source weights are compared with the verified leaderboard checkpoint before applying any mixture edit. [Leaderboard snapshot](results/leaderboard-start.json)

## Lessons from earlier parents

- Query sharpening on 917d6b4f improved the official public MATH500 and BBH results. The public macro rose from 47.33 to 47.36, but private fell from 54.70 to 53.60. Losses in IFEval, HumanEval and MMLU-Pro accompanied the reasoning gains. [Research PR 105](https://github.com/ilblackdragon/deltaleague/pull/105)
- Sharpening only the unrotated query coordinates produced the most consistent local macro signal on the old parent (+1.14 and +0.95). On 78 additional reasoning items both candidates answered 32 correctly. This is the control for transfer to the new parent.
- Adjusting recurrent decay globally within a layer range was costly. Middle-layer slowdown lost 7.47 macro points. The selective write-gate patch showed a mathematical gain, with 37 versus 33 correct out of 64 across disjoint panels, but BBH totaled 48 versus 50 out of 135. Three broad panels averaged −0.19. Two extra HumanEval passes out of four can move a local 18-group macro by 2.78 points.
- Truncation rescue and final-answer-only resampling had little effect on earlier parents. The state-dependent gain edit changed sampling as intended but scored 53.50 privately against 54.30 for its parent. Mechanism checks and benchmark checks answer different questions.
- Arithmetic SFT reduced held-out solution NLL from 1.7851 to 0.8180. In paired free generation it lost four net GSM8K answers (3/24 versus 7/24) and six IFEval answers (1/8 versus 7/8).
- A softened listwise adapter gained ten net correct answers on 576 held-out science questions. On the independent 204-item general screen it scored 44.50 versus 48.51, a loss of 4.01 macro points. Specializing a domain requires measuring the cost to the other benchmarks.

## New-parent comparison

The official new parent ties a6eb9f13 in public MATH500 and BBH and exceeds both source models in HumanEval and multi-turn BFCL. The first edit reapplies the retained unrotated-query operation to the new weights. Two reciprocal edits test changing the source mixture only in attention output projections of layers 8 through 22. A fourth edit adjusts only MLP down projections towards the old leader. The three mixture edits protect any weights that disagree with the rounded arithmetic midpoint. No training examples or private evaluations are used to fit these edits.

The [registered protocol](protocol.json) fixes all edits, model selection and paired evaluation before scoring. The baseline for each comparison is the verified 96b67023 checkpoint. All four candidates receive the 18-group development panel. A single frozen finalist receives a disjoint 18-group comparison and new MATH500/BBH questions. The published scores of the parent remain separate from the local MLX estimates.

## Local reproduction

Run these commands from this research directory on the Mac evaluation bed. The T9 experiment directory must sit alongside the three verified source models in `models/` and the prior suite and plan files referenced by the scripts. The evaluation runner checks the SHA-256 of the shared paired runner before loading it. Use a separate output directory for a new run.

```sh
source /Volumes/T9/ai-models/deltaleague/public-eval/env.sh
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
run_dir=/Volumes/T9/ai-models/deltaleague/delta-experiments/leader96-mixture-reproduction
delta_bin=/Users/jc/delta/deltaleague/.venv/bin
mkdir -p "$run_dir"
"$delta_bin/python" build.py "$run_dir"
"$DELTA_EVAL_ROOT/.venv/bin/python" fresh_panel.py "$run_dir"
for submission in "$run_dir"/*/submission; do
    "$delta_bin/delta" validate "$submission" > "$(dirname "$submission")/validation.log" || exit 1
done
"$DELTA_EVAL_ROOT/.venv/bin/python" -u evaluate_round.py "$run_dir"
```

## Results on 96b67023

The four candidates scored 46.37 for query transfer, 48.97 for attention toward a6eb9f13, 47.29 for attention toward 917d6b4f and 47.32 for MLP toward 917d6b4f on the 204-item development panel. The parent scored 47.48. Only attention toward a6eb9f13 advanced. Its net gains were one IFEval item and two MMLU-Pro items.

On a disjoint 204-item confirmation panel the finalist scored 48.12 versus 46.16. It gained two BBH, two IFEval, one MMLU-Pro and one GSM8K answer, and lost one single-turn BFCL answer. The paired 90% intervals for the macro differences were [−0.01, +3.21] on development and [−0.37, +4.30] on confirmation. On the 94 additional questions, MATH500 improved from 22/40 to 23/40 (one gain, zero losses). BBH fell from 23/54 to 21/54 (one gain, three losses). The local broad gain replicated, while the prespecified joint reasoning objective failed. [Machine-readable decision](results/decision.json)

## Transfer to d0120ae6

During the first run udirobert's three-source mixture reached a private score of 54.90. The model title gives 40% 917d6b4f, 40% a6eb9f13 and 20% b338518d. The [single-candidate transfer protocol](transfer/protocol.json) was registered after the first finalist's broad confirmation. The previous BF16 finalist minus its verified previous parent is added in FP32 to the new organizer checkpoint, then rounded once. This preserves the third source's contribution outside the transferred edit and defines an exact change on the new parent. The two general panels use the same question IDs with different generation seeds. A further 94 questions exclude all the earlier reasoning panels. Parent and candidate generate every response again in the new comparison.

For local reproduction, place the verified official d0120ae6 model in the same models directory. From this research directory and the same shell environment as above, run the following after completing the first finalist build in the original T9 experiment directory.

```sh
run_dir=/Volumes/T9/ai-models/deltaleague/delta-experiments/leaderd012-transfer-reproduction
mkdir -p "$run_dir"
"$delta_bin/python" build_transfer.py "$run_dir"
"$DELTA_EVAL_ROOT/.venv/bin/python" fresh_panel.py "$run_dir" transfer/protocol.json
"$delta_bin/delta" validate "$run_dir/attention-transfer/submission" > "$run_dir/attention-transfer/validation.log"
"$DELTA_EVAL_ROOT/.venv/bin/python" -u evaluate_round.py "$run_dir" transfer/protocol.json
```

The full transfer changed 537,508 values. On d0120ae6 its local development macro was 45.46 versus 43.98 (+1.48) and its confirmation macro was 46.44 versus 47.92 (−1.48). The two changes cancel to the displayed precision. New questions gave 16/40 versus 17/40 on MATH500 and 21/54 versus 21/54 on BBH. The BBH tie included four gains and four losses. [Transfer decision](transfer/results/decision.json)

After the two broad comparisons and before either fresh candidate comparison had a result, we registered a follow-up that restricts the same transfer to the three full-attention output projections (layers 11, 15 and 19). It changes 90,129 values. Each child answer is generated independently. The d0120ae6 parent reports are reused after checking their hashes, item IDs, input settings and model identity. The broad panels informed the choice of ablation, while the independent reasoning comparison remained blinded when the ablation was fixed. [Ablation protocol](ablation/protocol.json)

To reproduce the ablation on the same evaluation bed, use the verified three-source run and its native receipts in `leaderd012-transfer-v1`.

```sh
run_dir=/Volumes/T9/ai-models/deltaleague/delta-experiments/leaderd012-full-attention-reproduction
mkdir -p "$run_dir"
"$delta_bin/python" build_transfer.py "$run_dir" ablation/protocol.json
"$delta_bin/delta" validate "$run_dir/full-attention-transfer/submission" > "$run_dir/full-attention-transfer/validation.log"
"$DELTA_EVAL_ROOT/.venv/bin/python" -u evaluate_ablation.py "$run_dir"
```

The three-tensor ablation scored 46.55 versus 43.98 on development (+2.57, paired 90% interval +0.28 to +5.14) and 48.32 versus 47.92 on confirmation (+0.40, interval −0.22 to +1.21). The confirmation gain came from one MMLU-Pro answer. On 94 fresh questions it lost one previously correct MATH500 answer (16/40 versus 17/40) and two previously correct BBH answers (19/54 versus 21/54), with zero gains in either group. We retain the three-tensor patch for its broad-panel signal. Neither d0120ae6 edit met the registered joint reasoning and broad replication target. [Ablation decision](ablation/results/decision.json)

The official leaderboard still ranked d0120ae6 first at the end of the experiment. At that point all candidate results were local. [Closing leaderboard](results/leaderboard-close.json)

## Official submission

Sun-Jc submitted the three-layer full-attention patch on 25 November 2026 at 01:29 UTC. [Submission 4a2abfda](https://deltaleague.ai/submissions/4a2abfda-eb1e-458c-be5c-b744d5902576/) · [Research PR 108](https://github.com/ilblackdragon/deltaleague/pull/108)

The organizer verified model `63fa172a-3e69-49cf-b667-65f2db8810b3` on 25 November at 09:02 UTC. Official public validation was 47.46 versus 48.50 for d0120ae6. The official private test score was 53.40 ± 1.58 versus 54.90, a difference of −1.50 and tied rank 11 at verification. Public MATH500 rose from 48.44 to 56.25, IFEval rose from 76.56 to 85.94 and MMLU-Pro rose from 34.29 to 40.00. BBH fell from 42.59 to 33.33 and HumanEval from 37.50 to 15.63. The platform reports shared-fleet cost attribution and does not provide per-submission GPU usage. [Verified scores](ablation/results/official-submission.json)

To reproduce just the submitted payload and manifest, install the project CLI and its weight-edit dependencies and run the following in this research directory. Choose a filesystem with room for four official model versions and a temporary full model. The script verifies every official file and both output SHA-256 hashes. Here `models` is the destination directory for the four downloads.

```sh
models=/path/to/models
delta model pull 917d6b4f-fea3-432c-9264-e4f89e45890a --output "$models/917d6b4f"
delta model pull a6eb9f13-47fb-4755-bd08-268a86307674 --output "$models/a6eb9f13"
delta model pull 96b67023-ce48-4cfa-be8c-8db86694f154 --output "$models/96b67023"
delta model pull d0120ae6-20e2-4cd8-a73e-136470311df5 --output "$models/d0120ae6"
python reconstruct_ablation.py "$models" /path/to/new-submission
delta validate /path/to/new-submission
```
