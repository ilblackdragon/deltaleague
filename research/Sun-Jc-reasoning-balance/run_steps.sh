#!/bin/bash
set -eo pipefail
scripts="$(cd "$(dirname "$0")" && pwd)"
exp=/Volumes/T9/ai-models/deltaleague/delta-experiments
out="${1:?Provide a fresh or resumable T9 experiment directory}"
original="${2:?Provide the completed three-arm experiment directory}"
arm="$out/step-pairs"
cli=/Users/jc/delta/deltaleague/.venv/bin
exporter=/Users/jc/delta/worktrees/arithmetic-lora/research/Sun-Jc-arithmetic-lora/export.py
source "$exp/../public-eval/env.sh"
export DELTA_DATASET_SOURCES="$(cat "$DELTA_EVAL_ROOT/dataset-sources.json")"
evalpy="$DELTA_EVAL_ROOT/.venv/bin/python"
template="$original/seed-three/plan.json"
"$evalpy" "$scripts/step_pairs.py" "$arm"
if [[ ! -f "$arm/training.json" ]]; then
    "$DELTA_EVAL_ROOT/.venv-mlx/bin/python" -u "$scripts/train_steps.py" "$arm" "$template"
fi
if [[ ! -f "$arm/export.json" ]]; then
    "$cli/python" "$exporter" "$arm" "$exp/models/917d6b4f" "$template" \
        --parent-id 917d6b4f-fea3-432c-9264-e4f89e45890a --layers 16 20 --arm step-pairs
fi
"$cli/delta" validate "$arm/submission" > "$arm/validation.log"
cp "$arm/export.json" "$arm/build.json"
cp "$original/fresh-suite.json" "$original/fresh-selection.json" "$out/"
"$evalpy" -u "$scripts/evaluate.py" "$out" --study step-pairs
"$evalpy" -u "$scripts/fresh_panel.py" "$out" audit
