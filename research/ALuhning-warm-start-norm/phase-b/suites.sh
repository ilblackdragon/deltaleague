#!/usr/bin/env bash
set -euo pipefail
r2=${R2_CODE_DIR:-"$HOME/deltaleague-exp/r2"}
out=$1
shift
mkdir -p "$out"
py="$HOME/.venvs/train/bin/python"
public="$PWD/.venv-public/bin/python"
export DELTA_WASI_ROOT="$PWD/.wasi"
url=http://127.0.0.1:18000/v1
for model in "$@"; do
    name=${model##*/}
    report="$out/public-$name"
    [[ ! -e $report ]]
    PYTHONPATH="$PWD/pylib-fla:$PWD/pylib:$PWD" "$py" "$r2/serve.py" "$model" --sequences 32
    if [[ $model == model-orig ]]; then
        hash=$(python3 -c 'import json; print(json.load(open("model-orig/delta-model.json"))["content_hash"])')
    else
        hash=$(sha256sum "$model/model.safetensors" | cut -d' ' -f1)
    fi
    if ! env -u PYTHONPATH "$public" suite_debug.py "$url" policy "$PWD/model-orig" "$hash"         "$report" validation.json evaluation.json >"$out/$name.log" 2>&1; then
        tail -65 "$out/$name.log"
        case $name in
            model-3in1-n12|model-3in1-n14)
                podman stop --time 20 dl-r2-vllm
                echo "Incomplete public suite for optional Phase A edit $name"
                continue
                ;;
            *) exit 1 ;;
        esac
    fi
    podman stop --time 20 dl-r2-vllm
    echo "Public suite complete for $name"
done
