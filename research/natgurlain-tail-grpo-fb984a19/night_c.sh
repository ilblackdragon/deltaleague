#!/bin/bash
# Part C: resume tail-only training from the saved state to step 16, probe steps 16 and 12, push results.
cd "$(dirname "$0")"; . ./gpu_lib.sh
GPU_MEM=0.25 LOGIT_CHUNK=256 attempt 22000 4 $PY grpo.py parent runs/tail --pool-file mined/tail.jsonl --resume \
  --steps 16 --prompts 128 --group 8 --save-at 4,8,12,16 --train-max-tokens 6144 --match-sampler --len-shape 3000 \
  --seed 20261115 && rm -f runs/tail/resume.pt runs/tail/resume.tmp || echo "training did not finish; resume state kept"
for s in 16 12; do
  [ -f runs/tail/step$s/model.safetensors ] || continue
  probe tail_s$s runs/tail/step$s && $PY cmp.py probe/leader.jsonl probe/tail_s$s.jsonl | tee probe/cmp_tail_s$s.txt
done
echo "[$(date +%H:%M)] part C finished"
cd .. && git add -A && git -c user.name="natgurlain" -c user.email="natgurlain@users.noreply.github.com" commit -q -m "Tail-only GRPO: steps 10-16 and probes of steps 12 and 16

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" && git push -q origin main && echo "pushed"
