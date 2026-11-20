#!/bin/bash
# Part E: Fair Cast GRPO (stochastic-rounding export + loss on the served copy; 2610.41073, 2611.13545).
# 60 steps of 32 prompts x 8 on the mixed pool (mined tail + 25% other mined prompts), checkpoints every 10 steps.
cd "$(dirname "$0")"; . ./gpu_lib.sh
GPU_MEM=0.25 LOGIT_CHUNK=256 attempt 22000 6 $PY grpo.py parent runs/fc --pool-file mined/mix.jsonl --resume --fair-cast \
  --steps 60 --prompts 32 --group 8 --save-at 10,20,30,40,50,60 --train-max-tokens 6144 --match-sampler --len-shape 3000 \
  --seed 20261116 && rm -f runs/fc/resume.pt runs/fc/resume.tmp || echo "training did not finish; resume state kept"
echo "[$(date +%H:%M)] part E training finished"
