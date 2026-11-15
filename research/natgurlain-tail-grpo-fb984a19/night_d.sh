#!/bin/bash
# Part D: replicate the step 16 result on a second, fresh set of 1,000 held-out prompts.
cd "$(dirname "$0")"; . ./gpu_lib.sh
X="--prompts 1000 --seed 3 --k 4 --exclude probe/leader.jsonl --exclude mined/leader.jsonl"
GPU_MEM=0.75 attempt 18500 6 $PY mine.py parent probe/leader_p2.jsonl $X || exit 1
GPU_MEM=0.75 attempt 18500 6 $PY mine.py runs/tail/step16 probe/tail_s16_p2.jsonl --k 4 --items probe/leader_p2.jsonl || exit 1
$PY cmp.py probe/leader_p2.jsonl probe/tail_s16_p2.jsonl | tee probe/cmp_tail_s16_p2.txt
echo "[$(date +%H:%M)] part D finished"
