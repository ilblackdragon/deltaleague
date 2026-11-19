#!/usr/bin/env bash
# usage: scripts/pubeval.sh MODEL_DIR NAME   -> runs/pub_NAME/result.json
set -euo pipefail
cd "$(dirname "$0")/.."
M=$(realpath "$1"); N=$2; PORT=${PORT:-8011}
export VLLM_USE_FLASHINFER_SAMPLER=0
.venv/bin/vllm serve "$M" --served-model-name candidate --port $PORT --language-model-only \
  --dtype bfloat16 --max-model-len 34816 --max-num-seqs 128 --gpu-memory-utilization 0.82 \
  --no-enable-log-requests > runs/pub_$N.server.log 2>&1 &
SP=$!; trap "kill $SP 2>/dev/null; wait $SP 2>/dev/null" EXIT
until curl -sf localhost:$PORT/v1/models >/dev/null; do sleep 5; kill -0 $SP || { tail runs/pub_$N.server.log; exit 1; }; done
H=$(cat "$M"/*.safetensors | sha256sum | cut -c1-64)
rm -rf runs/pub_$N
DELTA_WASI_ROOT=$PWD/pe/wasi HF_ALLOW_CODE_EVAL=1 .venv-pe/bin/delta public-eval run --endpoint http://localhost:$PORT/v1 \
  --tokenizer "$M" --content-hash $H --output runs/pub_$N --manifest pe/validation.json \
  --evaluation-config pe/evaluation.json
