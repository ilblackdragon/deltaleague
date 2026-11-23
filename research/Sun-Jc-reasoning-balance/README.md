# Joint reasoning and general transfer

Compare middle-to-late attention concentration and a third GRPO seed blend against official version 917d6b4f-fea3-432c-9264-e4f89e45890a. The two attention edits scale the effective query RMSNorm gain on full-attention layers 11, 15 and 19. The seed blend uses weights 1/2, 1/4 and 1/4 for eff1e1b7, a6eb9f13 and fdfd46a4 on layers 0–22. All edits retain the leading model's last layer, token embedding and output normalization. The inputs are public versioned checkpoints. There is no task-specific fitting.

The selection and replication rules are in `protocol.json`. Every candidate receives the same 18 native local benchmark groups. The scorer measures MATH500 with math verification and BBH with its native filtered exact match. A development group has eight mathematics questions or one question per each of 27 BBH tasks. A complete score averages the 18 benchmark scores equally. The previous leader run is reused only after reconstructing its scores from native reports and checking all model, suite and runtime identities.

## Results on November 23, 2026

Keep 917d6b4f as the working baseline. All four candidates scored below it on the local 204-item, 18-benchmark development subset. Each benchmark has equal weight. MATH500 and BBH entries are numbers of correct answers. The interval is a paired 90% bootstrap interval for the difference in macro score under this one generation seed.

| Checkpoint or edit | Macro | Change | 90% interval | MATH500 / 8 | BBH / 27 |
| --- | ---: | ---: | --- | ---: | ---: |
| Official leader 917d6b4f | 48.51 | — | — | 5 | 11 |
| Stronger attention, ×1.08 | 45.25 | −3.26 | [−7.94, +1.51] | 3 | 9 |
| Softer attention, ×0.92 | 43.92 | −4.59 | [−8.71, −0.44] | 5 | 12 |
| Third GRPO seed blend | 44.71 | −3.80 | [−8.23, +0.57] | 3 | 12 |
| Paired reasoning-step LoRA | 46.11 | −2.40 | [−6.43, +1.69] | 4 | 9 |

On the 35 disjoint reasoning items, the parent scored 5/8 on MATH500 and 7/27 on BBH. Stronger attention matched the parent's mathematics answers and scored 12/27 on BBH (eight wins, three losses). The reasoning-step adapter matched the mathematics answers and scored 5/27 on BBH (four wins, six losses). The parent's two fresh runs returned identical responses on every item. Neither experiment qualified for the larger target panel or full 18-benchmark replication under its preregistered rules.

The repeated sign change for stronger attention on BBH (two fewer correct on development, five more on replication) limits the case for that edit. In the separate reasoning-step experiment, MBPP improved by two questions and MMLU-Pro improved by two, while HumanEval lost three of four, illustrating the influence of small equally weighted groups. Local results describe the sampled benchmarks. The official private score of the baseline was 54.70, and the leaderboard's top four versions and scores were unchanged at 15:46 Singapore time.

The combined results, paired intervals, native-report checks and parent repeatability are in [the evaluation summary](results/summary.json). Individual benchmark results are in the adjacent development reports. Raw reports and the reconstructed models are on T9 in `delta-experiments/reasoning-balance-v1` and `delta-experiments/reasoning-steps-v1`.

## Exploratory official evaluation

Sun-Jc selected one official evaluation of the stronger-attention patch after completing the local comparison. The disjoint BBH result supplied a positive signal, with eight wins and three losses, and all eight MATH500 answers retained their original scores. The negative 18-benchmark development result and the change in direction on BBH make this an exploratory submission. The original decision to retain the parent as the working baseline still applies.

The parent version is `917d6b4f-fea3-432c-9264-e4f89e45890a`. The selected sparse patch has SHA-256 `75ab09205836a490b84546eefbe97a1fc220a6340ce1106b39fdd5bc737ba9ce` and occupies 8,090 bytes. Delta League received [submission 8cd9373f](https://deltaleague.ai/submissions/8cd9373f-c20d-439f-889e-d9d7de84d716/) at 09:07 UTC on November 23, 2026, linked to [research PR 105](https://github.com/ilblackdragon/deltaleague/pull/105). Official scoring completed on November 23, 2026. The candidate ranked joint fifth at verification with 53.60 on the private test. The parent ranked first with 54.70. Official public validation was 47.36 for the candidate and 47.33 for the parent. Both target benchmark scores rose on public validation. MATH500 moved from 50.00 to 53.13 and BBH from 27.78 to 37.04. The private score was 1.10 points below the parent. The [official result record](results/official-query-sharp.json) contains the version IDs, evaluation configuration and public benchmark breakdown.

## Reproduction

Use the T9 Python, MLX and evaluation environments from the Mac evaluation guide. The laboratory batch server, baseline verifier and analysis come from the GRPO calibration branch at commit 450bf5a. Retain the previous verified leader run, the disjoint suite and the pinned native data on T9. Run from this research folder with a new directory under delta-experiments.

```sh
root=/Volumes/T9/ai-models/deltaleague
exp="$root/delta-experiments"
out="$exp/reasoning-balance-reproduction"
bin=/Users/jc/delta/deltaleague/.venv/bin
mkdir "$out"
"$bin/delta" model pull fdfd46a4-3a55-4277-91c1-70c739d6139b --output "$exp/models/fdfd46a4"
for arm in query-sharp query-soft seed-three; do
    "$bin/python" build.py "$out" "$arm" || exit 1
    "$bin/delta" validate "$out/$arm/submission" > "$out/$arm/validation.log" || exit 1
done
source "$root/public-eval/env.sh"
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
"$DELTA_EVAL_ROOT/.venv/bin/python" -u evaluate.py "$out"
```

An additional target panel freezes 24 MATH500 questions and two questions in each of the 27 BBH tasks outside the official public manifest. The panel is opened only if the selected candidate qualifies for full replication or both targets strictly improve in the disjoint replication. The feature weights and selection rule stay fixed. Run `fresh_panel.py "$out" prepare` before reading joint development outcomes and `fresh_panel.py "$out" audit` after replication, using the same evaluator environment.

The leaderboard check on November 23, 2026 found the same leader, with public validation 47.33 and private test 54.70. Local main was fast-forwarded to the organizer's 9e3435b. The existing evaluation environment remains pinned at b9db793 for comparison with the recorded baseline. The native text, coding, tools and suite modules used here have identical source at both revisions.

## Counterfactual reasoning steps

A separate fourth edit trains on six synthetic problem families. Each training pair changes a condition so the correct next token changes. The 96 training pairs and 24 diagnostic pairs have 240 unique contexts. An independent computation from the question texts verified all labels, and the leader tokenizer verified that each choice pair differs only in its last token. The two trained projections are in language layers 16 and 20. The loss fits the correct token against the matched alternative and penalizes divergence from the leader's next-token distribution.

The fourth arm has its own preregistered protocol, development result and replication decision. It uses the same 18-group weighting, item sets, native scoring and generation configuration as the preceding experiment. To reproduce the fixed fit and evaluation, run the following command after the three-arm study completes, using a separate output directory on T9.

```sh
./run_steps.sh "$exp/reasoning-steps-reproduction" "$out"
```
