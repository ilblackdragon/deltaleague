# Shared helpers: wait for free GPU memory (other projects share this card) and retry interrupted jobs.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NLTK_DATA=${NLTK_DATA:-$PWD/../.nltk_data} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../.venv-grpo/bin/python
wait_gpu() {  # wait_gpu MIB : block until that much memory is free on three checks in a row
  local need=$1 ok=0 waited=0
  while [ $ok -lt 3 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    if [ "$free" -ge "$need" ]; then ok=$((ok+1)); sleep 10; else
      [ $((waited % 30)) -eq 0 ] && echo "[$(date +%H:%M)] waiting for GPU: ${free} MiB free, need ${need}"
      ok=0; waited=$((waited+1)); sleep 60; fi
  done
}
attempt() {  # attempt NEED_MIB TRIES cmd... : run, and on failure wait for memory and run again
  local need=$1 tries=$2; shift 2
  for i in $(seq 1 "$tries"); do
    wait_gpu "$need"
    echo "[$(date +%H:%M)] start (try $i): $*"
    if "$@"; then echo "[$(date +%H:%M)] ok: $1 $2 $3"; return 0; fi
    echo "[$(date +%H:%M)] failed (try $i)"; sleep 60
  done
  return 1
}
probe() {  # probe NAME MODEL_DIR
  local src="--prompts 1000 --seed 1"
  [ "$1" != leader ] && src="--items probe/leader.jsonl"
  GPU_MEM=0.75 attempt 18500 6 $PY mine.py "$2" "probe/$1.jsonl" --k 4 $src
}
