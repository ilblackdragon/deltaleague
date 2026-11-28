#!/bin/bash
# JSON-files contract fix, stage A: fc step 60 samples its own
# answers on the JSON-contract prompts of MBPP train/validation tasks (and on the replay prompts). Generation only.
cd "$(dirname "$0")"; . ./gpu_lib.sh
t0=$(date +%s)
GPU_MEM=0.75 attempt 18500 3 $PY mine.py runs/fc/step60 ../jsonfix/gen_fc60.jsonl --k 4 --items ../jsonfix/gen_items.jsonl --save-text || exit 1
GPU_MEM=0.75 attempt 18500 3 $PY mine.py runs/fc/step60 ../jsonfix/replay_fc60.jsonl --k 2 --items ../jsonfix/replay_items.jsonl --save-text || exit 1
echo "[$(date +%H:%M)] jsonfix stage A finished after $(( $(date +%s) - t0 ))s"
