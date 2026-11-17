#!/bin/bash
# Format-and-finishing census. MEASUREMENT ONLY, no training.
# Leader and fc step 60, 600 prompts x 4 samples with text (census_build.py), then about 366 single-turn tool-calling
# items through the official harness. Scoring is done afterwards on CPU (census_score.py). Budget: about 1.3 GPU-hours.
cd "$(dirname "$0")"; . ./gpu_lib.sh
t0=$(date +%s)
for c in parent:leader runs/fc/step60:fc_s60; do
  d=${c%%:*}; n=${c##*:}
  GPU_MEM=0.75 attempt 18500 3 $PY mine.py $d ../census/$n.jsonl --k 4 --items ../census/items.jsonl --save-text || { echo "[$(date +%H:%M)] census: generation failed for $n"; exit 1; }
done
el=$(( $(date +%s) - t0 )); echo "[$(date +%H:%M)] census: text slices done after ${el}s"
if [ $el -gt 4200 ]; then echo "[$(date +%H:%M)] census: budget guard, tool-calling slice skipped"; exit 0; fi
bfcl() {  # bfcl NAME MODEL_DIR
  local name=$1 dir; dir=$(realpath "$2"); local out=../census/bfcl_$name; mkdir -p "$out"
  local hash; hash=$(sha256sum "$dir/model.safetensors" | cut -d' ' -f1)
  wait_gpu 18500
  ../.venv-grpo/bin/vllm serve "$dir" --served-model-name candidate --host 127.0.0.1 --port 8000 --language-model-only \
    --dtype bfloat16 --max-model-len 34816 --max-num-seqs 64 --gpu-memory-utilization .70 --no-enable-log-requests \
    > ../logs/vllm_census_$name.log 2>&1 &
  local srv=$!
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8000/v1/models && break; sleep 10; done
  echo "[$(date +%H:%M)] census bfcl $name: server up"
  printf '%s\n' "0 70" "240 256" "256 326" "456 526" "656 726" "856 926" | xargs -P 3 -L 1 bash -c '
    d="census/bfcl_'"$name"'/${0}_${1}"; cd .. && rm -rf "$d" && DELTA_WASI_ROOT=.wasi NLTK_DATA=.nltk_data .venv-public/bin/delta public-eval benchmark \
      --protocol broad/proto_bfcl_single_turn.json --endpoint http://127.0.0.1:8000/v1 --tokenizer "'"$dir"'" --content-hash "'"$hash"'" \
      --output "$d" --start "$0" --end "$1" > "logs/census_bfcl_'"$name"'_${0}.log" 2>&1 || echo "shard failed: $0 $1"'
  kill $srv; wait $srv 2>/dev/null; sleep 20
  echo "[$(date +%H:%M)] census bfcl $name finished: $(ls "$out"/*/result.json 2>/dev/null | wc -l) of 6 shards"
}
bfcl leader parent
bfcl fc_s60 runs/fc/step60
echo "[$(date +%H:%M)] census finished after $(( $(date +%s) - t0 ))s"
