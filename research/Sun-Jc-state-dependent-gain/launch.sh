#!/bin/sh
# Start P single-threaded gen.py workers on one job file and wait for all of them.
# usage: launch.sh MODEL_FP32_DIR JOBS OUT_DIR P [extra gen.py args...]
model=$1; jobs=$2; out=$3; P=$4; shift 4
mkdir -p "$out"
export OMP_NUM_THREADS=1
k=0
while [ "$k" -lt "$P" ]; do
  "${PYTHON:-python}" -u "$(dirname "$0")/gen.py" --model "$model" --jobs "$jobs" --out "$out" --shard "$k/$P" "$@" 2>> "$out/log-$k.txt" &
  k=$((k + 1))
done
wait
echo "all $P workers finished: $(cat "$out"/shard-*.jsonl | wc -l) completions in $out"
