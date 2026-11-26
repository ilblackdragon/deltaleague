# Routing components of the three-source leader

Sun-Jc and Codex compare four tensor grafts from the public two-source mixture 96b67023 into the official d0120ae6 parent. The two-source model has a higher official public MATH500 score (54.69 versus 48.44) and ties the parent on BBH (42.59). The previous attention-output patch lost three answers on fresh reasoning questions. This experiment tests input routing and late computations. The four edits and advancement rule are in [the protocol](protocol.json).

The two 204-item panels use the same seeds and verified d0120ae6 parent responses as the previous study. The new reasoning panel has 40 MATH500 questions and two BBH questions per subtask. Model selection uses public data only. Raw generated answers and weights are kept on T9.

## Selected candidate

The MLP channel graft qualified under its original 204/332/94 protocol. Its 18-task mean rose from 43.98 to 45.16 in development and from 50.77 to 52.24 in confirmation. The separate held-out MATH500 score rose from 14/40 to 16/40 and BBH rose from 16/54 to 17/54. The two math changes and one BBH change were all gains in the paired comparison. An additional overall panel and the later mixture controls were conditional on this candidate failing. They did not contribute to its selection. Many earlier edits were screened on the broad development data. The fresh result rests on three changed answers under one generation seed. Official private performance requires organizer scoring. [Registered protocol](mlp-channels/protocol.json) · [Fresh reasoning results](mlp-channels/results/fresh-mlp-channel-graft.json) · [Selection decision](mlp-channels/results/decision.json)

The organizer received the submission on 2026-11-26 at 14:51 UTC under the private1000-v6-thinking8k-20261111 evaluation. [Official submission](https://deltaleague.ai/submissions/1ad1e001-24f1-45e4-adda-3f683d310b89/) · [Research PR](https://github.com/ilblackdragon/deltaleague/pull/112)

## Local scores

Each value below is the candidate's 18-task mean minus the parent's mean, in percentage points. Development and confirmation use disjoint questions. The parent scored 43.98 and 47.92 on the respective panels. A dash indicates that the candidate did not advance to that panel.

| Edit on d0120ae6 | Development (204) | Confirmation (204) |
| --- | ---: | ---: |
| Full attention Q/K/V from 96b67023 | +0.49 | — |
| Recurrent content from 96b67023 | +1.18 | +0.69 |
| Late MLP inputs from 96b67023 | +0.99 | — |
| Late blocks from 96b67023 | +1.38 | +2.69 |
| Late blocks plus recurrent content | −0.01 | +0.89 |
| Fill missing seed-3 changes | +2.65 | — |
| Add half the seed-3 change | +3.54 | −2.69 |
| Revert changes opposed to seed 3 | +3.76 | −0.79 |

On two additional reasoning panels totaling 80 MATH500 and 108 BBH items, the late-block edit tied its parent on MATH500 and gained nine correct BBH answers. The recurrent-content edit gained one on each task. Neither edit met the prespecified two-cohort reasoning criterion. On a third 18-task panel of 332 items, late-block scored 50.55 against 50.77 for its parent. The paired 90% interval for the difference was −2.26 to +1.90 points. These local results use sampled generation and have substantial uncertainty. [Routing](results/decision.json) · [Combined edit](union/results/decision.json) · [Two-cohort reasoning results](replication/results/decision.json) · [Late-block generalization](generalization/results/decision.json) · [Seed-3 edits](residual/results/decision.json)

On a separate 332-item panel, recurrent-content scored 43.21 against the parent's 43.27. The paired 90% interval for the difference was −0.80 to +0.69. MATH500 was tied at 12/24 and BBH fell from 12/27 to 11/27. This ruled out the patch under its registered three-panel criterion. [Recurrent-content generalization](recurrent-generalization/results/decision.json)

The consensus edit raised all underrepresented decoder projection changes shared by the seed-1/2 and seed-3 donors to the nearer donor value in layers 0–22. It gained 1.97 and 0.28 points on the two 204-item panels. On 94 new questions it gained three BBH answers (18/54 versus 15/54) and lost one MATH500 answer (21/40 versus 22/40). The recurrent-only variant gained 1.88 points in development and lost 0.90 in confirmation. [Consensus study](consensus/results/decision.json)

The final-channel transplant copied three final-layer MLP channels and the final RMSNorm from bde3d8e4, changing 10,236 values. Its development gain of 3.40 points reversed on the 332-item panel, where it scored 49.51 against 50.77. MATH500 stayed at 12/24 and BBH fell from 11/27 to 8/27. The separate 94-item holdout remained unused for this edit. [Final-channel study](final-channel/results/decision.json)

The MLP channel graft copies the entire gate row, up row and down column of 448 neurons per layer from 96b67023 to d0120ae6 in layers 0–22. The selector ranks neurons by the number of differing weights and resolves ties by channel index. The sparse patch replaces 700,954 values. Its 18-task mean increased from 43.98 to 45.16 on 204 development items and from 50.77 to 52.24 on 332 disjoint confirmation items. The paired 90% interval for the second difference was +0.20 to +2.96 points. Confirmation MATH500 went from 12/24 to 13/24 and BBH from 11/27 to 10/27. [Graft protocol](mlp-channels/protocol.json) · [Development](mlp-channels/results/development-mlp-channel-graft.json) · [Confirmation](mlp-channels/results/confirmation-mlp-channel-graft.json)

## Reproduction

With the project CLI, Torch and safetensors installed, run the following from this research directory. The builder checks every official file in each model directory, reconstructs the edit from those inputs and verifies the exact manifest and payload hashes against the registered build record.

```sh
models=/path/to/models
delta model pull d0120ae6-20e2-4cd8-a73e-136470311df5 --output "$models/d0120ae6"
delta model pull 96b67023-ce48-4cfa-be8c-8db86694f154 --output "$models/96b67023"
python reproduce.py "$models" protocol.json late-block /path/to/new-submission
delta validate /path/to/new-submission
```

The same public model downloads reproduce the MLP channel graft. Its payload SHA-256 is `fd1b21acb0d19a222b67b0f426eea3d6445392b2418e4ac4d1f032b55829e295`. The locally rebuilt payload and manifest matched the evaluated files byte for byte. [Artifact hashes](mlp-channels/results/builds.json)

```sh
python reproduce.py "$models" mlp-channels/protocol.json mlp-channel-graft /path/to/new-graft
delta validate /path/to/new-graft
```

On the Mac evaluation bed, the paired local comparison uses the pinned helpers from the previous experiment at local commit `e0b30ba` (same helper files in [PR 108](https://github.com/ilblackdragon/deltaleague/pull/108)). The baseline cache contains reports from the actual d0120ae6 checkpoint. The runner checks the model and suite hashes, sample IDs, generation settings and raw-report receipts before reuse. Both models generate answers independently for the fresh panel. The official configuration is `private1000-v6-thinking8k-20261111`. The 18-task panels use seeds 20261111 and 20261112, the fresh seed is recorded per protocol, and all use thinking enabled with 8192 tokens, temperature 1, top-p 0.95, top-k 20 and presence penalty 1.5.

```sh
source /Volumes/T9/ai-models/deltaleague/public-eval/env.sh
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
run_dir=/Volumes/T9/ai-models/deltaleague/delta-experiments/new-routing-run
delta_bin=/Users/jc/delta/deltaleague/.venv/bin
"$DELTA_EVAL_ROOT/.venv/bin/python" study.py fresh "$run_dir"
"$delta_bin/python" study.py build "$run_dir"
for submission in "$run_dir"/*/submission; do
    "$delta_bin/delta" validate "$submission" > "$(dirname "$submission")/validation.log" || exit 1
done
"$DELTA_EVAL_ROOT/.venv/bin/python" -u study.py evaluate "$run_dir"
```
