#!/usr/bin/env bash
set -euo pipefail
r2=$PWD/r2
src=$PWD/data/r2
out=$1
family=$2
resume_math=${3:-}
mkdir -p "$out"
py="$HOME/.venvs/train/bin/python"
export PYTHONPATH="$PWD/pylib-fla:$PWD/pylib:$PWD"
export DELTA_WASI_ROOT="$PWD/.wasi"
url=http://127.0.0.1:18000/v1

"$py" "$r2/serve.py" model-lora1
case "$family" in
math|mc|rehearsal)
    items="$src/$family.jsonl"
    if [[ $family == math && -n $resume_math ]]; then
        cp --reflink=auto "$resume_math/math-probe.jsonl" "$resume_math/math-scored.jsonl" "$out/"
        "$py" "$r2/r2_data.py" select "$items" "$out/math-scored.jsonl" "$out/math-selected.jsonl" \
            --selected-prefix "$resume_math/math-selected.jsonl"
    else
        "$py" "$r2/r2_data.py" probe "$items" "$out/$family-probe.jsonl" --samples 2
        "$py" "$r2/r2_data.py" score "$items" "$out/$family-probe.jsonl" "$out/$family-scored.jsonl" --keep-text
        "$py" "$r2/r2_data.py" select "$items" "$out/$family-scored.jsonl" "$out/$family-selected.jsonl"
    fi
    ;;
hints)
    "$py" "$r2/hint_data.py" gen "$src/math-hints.jsonl" "$out/math-repair.jsonl" "$url" --conc 128 --cap 8192
    ;;
*)
    exit 2
    ;;
esac
podman stop --time 20 dl-r2-vllm
