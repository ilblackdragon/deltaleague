#!/bin/bash
# Part B: mine the leader, keep the prompts with a learning signal, train on them, probe the checkpoints.
cd "$(dirname "$0")"; . ./gpu_lib.sh; mkdir -p mined
until [ -f probe/.part_a_done ]; do sleep 30; done
[ -s probe/leader.jsonl ] || { echo "no leader probe; stopping"; exit 1; }
GPU_MEM=0.75 attempt 18500 6 $PY mine.py parent mined/leader.jsonl --prompts 3000 --seed 2 --k 4 --exclude probe/leader.jsonl || exit 1
$PY select_tail.py mined/leader.jsonl mined/tail.jsonl 4096 | tee mined/tail_stats.json
GPU_MEM=0.25 LOGIT_CHUNK=256 attempt 22000 8 $PY grpo.py parent runs/tail --pool-file mined/tail.jsonl --resume \
  --steps 16 --prompts 128 --group 8 --save-at 4,8,12,16 --train-max-tokens 6144 --match-sampler --len-shape 3000 \
  --seed 20261115 && rm -f runs/tail/resume.pt runs/tail/resume.tmp || echo "training did not finish; resume state kept"
for s in 16 8 12; do
  [ -f runs/tail/step$s/model.safetensors ] || continue
  probe tail_s$s runs/tail/step$s && $PY cmp.py probe/leader.jsonl probe/tail_s$s.jsonl | tee probe/cmp_tail_s$s.txt
done
echo "[$(date +%H:%M)] part B finished"
