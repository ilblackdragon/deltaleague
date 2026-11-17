#!/bin/bash
# JSON-files contract fix, stage C: the frozen test and guards on ONE chosen checkpoint.
# usage: ./jsonfix_test.sh STEP [noprobe]
cd "$(dirname "$0")"; . ./gpu_lib.sh
s=$1; ck=runs/jsonfix/step$s; name=jf_s$s; dir=$(realpath "$ck"); t0=$(date +%s)
GPU_MEM=0.75 attempt 18500 3 $PY mine.py $ck ../census/$name.jsonl --k 4 --items ../census/items.jsonl --save-text || exit 1
echo "[$(date +%H:%M)] jsonfix test: census slices done after $(( $(date +%s) - t0 ))s"
hash=$(sha256sum "$dir/model.safetensors" | cut -d' ' -f1)
wait_gpu 18500
../.venv-grpo/bin/vllm serve "$dir" --served-model-name candidate --host 127.0.0.1 --port 8000 --language-model-only \
  --dtype bfloat16 --max-model-len 34816 --max-num-seqs 64 --gpu-memory-utilization .70 --no-enable-log-requests \
  > ../logs/vllm_$name.log 2>&1 &
srv=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8000/v1/models && break; sleep 10; done
echo "[$(date +%H:%M)] jsonfix test: server up"
mkdir -p ../census/bfcl_$name ../broad/$name
printf '%s\n' "bfcl_single_turn 0 70 census/bfcl_$name/0_70" "bfcl_single_turn 240 256 census/bfcl_$name/240_256" "bfcl_single_turn 256 326 census/bfcl_$name/256_326" \
  "bfcl_single_turn 456 526 census/bfcl_$name/456_526" "bfcl_single_turn 656 726 census/bfcl_$name/656_726" "bfcl_single_turn 856 926 census/bfcl_$name/856_926" \
  "mbpp 0 500 broad/$name/mbpp_0_500" | xargs -P 3 -L 1 bash -c '
    cd .. && rm -rf "$3" && DELTA_WASI_ROOT=.wasi NLTK_DATA=.nltk_data .venv-public/bin/delta public-eval benchmark \
      --protocol broad/proto_$0.json --endpoint http://127.0.0.1:8000/v1 --tokenizer "'"$dir"'" --content-hash "'"$hash"'" \
      --output "$3" --start "$1" --end "$2" > "logs/'"$name"'_${0}_${1}.log" 2>&1 || echo "shard failed: $0 $1 $2"'
kill $srv; wait $srv 2>/dev/null; sleep 20
echo "[$(date +%H:%M)] jsonfix test: harness slices done after $(( $(date +%s) - t0 ))s"
if [ "$2" != noprobe ]; then
  head -n 500 probe/leader_p2.jsonl > probe/p2_first500.jsonl
  GPU_MEM=0.75 attempt 18500 3 $PY mine.py $ck probe/${name}_p2h.jsonl --k 4 --items probe/p2_first500.jsonl \
    && $PY cmp.py probe/fc_s60_p2.jsonl probe/${name}_p2h.jsonl | tee probe/cmp_${name}_p2h_vs_fc60.txt
fi
echo "[$(date +%H:%M)] jsonfix stage C finished after $(( $(date +%s) - t0 ))s"
