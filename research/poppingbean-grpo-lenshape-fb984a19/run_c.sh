#!/bin/bash
# Option C: GRPO on fb984a19 with a gentle length penalty (correct answers over 3,000 tokens lose up to half
# their reward at the 6,144 cap). Same recipe as #58 otherwise; no reward-decline stop. Cap: JOB_HOURS (4).
set -x
cd /root/work
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/venv/bin:$PATH
PY=$PWD/venv/bin/python
JOB_END=$(( $(date +%s) + ${JOB_HOURS:-4}*3600 )); left() { echo $(( (JOB_END - $(date +%s)) / 60 )); }
VLLM_ALLOW_INSECURE_SERIALIZATION=1 GPU_MEM=0.35 timeout 6000 $PY grpo.py variants/parent runC \
  --steps 16 --save-at 8,16 --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 \
  --train-max-tokens 6144 --match-sampler --len-shape 3000 > results/trainC.log 2>&1
serve() { $PY -m vllm.entrypoints.openai.api_server --model variants/$1 --served-model-name candidate --port $2 \
  --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0 > results/serve_$1.log 2>&1 & srv=$!
  for _ in $(seq 180); do curl -sf localhost:$2/v1/models >/dev/null && return 0; sleep 5; done; return 1; }
for s in 16 8; do
  n=c$s; [ -s runC/step$s/model.safetensors ] || continue
  [ $(left) -gt 45 ] || { echo "job cap: skipping $n"; continue; }
  mkdir -p variants/$n; cp runC/step$s/* variants/$n/
  GPU_MEM=0.85 $PY probe.py variants/$n results/hold_$n.jsonl --n 256 --seed 7 > results/hold_$n.log 2>&1
  serve $n 8300 || { kill $srv; continue; }
  timeout $(( $(left) * 60 )) peval/bin/python pe_groups.py http://localhost:8300/v1 variants/$n results/pg_$n \
    dl_parent/validation.json dl_parent/evaluation.json > results/pg_$n.log 2>&1
  kill $srv; wait $srv 2>/dev/null
done
echo C-DONE
