#!/bin/bash
# JSON-files contract fix, stage B: the supervised fine-tune (settings fixed in advance, see README) and the dev-set samples
# for the three saved checkpoints. No test data is touched here.
cd "$(dirname "$0")"; . ./gpu_lib.sh
t0=$(date +%s)
LOGIT_CHUNK=256 attempt 22000 1 $PY sft_json.py runs/fc/step60 runs/jsonfix --data ../jsonfix/train.jsonl --steps 150 --lr 2e-6 \
  --warmup 10 --repaired 5 --valid 5 --replay 6 --save-at 50,100,150 --seed 20261117 || { echo "[$(date +%H:%M)] jsonfix training failed"; exit 1; }
echo "[$(date +%H:%M)] jsonfix training finished after $(( $(date +%s) - t0 ))s"
for s in 50 100 150; do
  GPU_MEM=0.75 attempt 18500 3 $PY mine.py runs/jsonfix/step$s ../jsonfix/dev_s$s.jsonl --k 4 --items ../jsonfix/dev_items.jsonl --save-text
done
echo "[$(date +%H:%M)] jsonfix stage B finished after $(( $(date +%s) - t0 ))s"
