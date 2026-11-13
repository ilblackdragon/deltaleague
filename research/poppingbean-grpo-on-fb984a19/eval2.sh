#!/bin/bash
# Evaluate parent (resuming its 7 finished groups) and step8 on the same public groups, plus step8 math probe.
cd /root/work; export PATH=$PWD/venv/bin:$PATH; PY=$PWD/venv/bin/python
serve() { $PY -m vllm.entrypoints.openai.api_server --model variants/$1 --served-model-name candidate --port $2 --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0 > results/serve2_$1.log 2>&1 & srv=$!; for _ in $(seq 180); do curl -sf localhost:$2/v1/models >/dev/null && return 0; sleep 5; done; return 1; }
[ -s results/hold_step8.jsonl ] || GPU_MEM=0.85 $PY probe.py variants/step8 results/hold_step8.jsonl --n 256 --seed 7 > results/hold_step8.log 2>&1
for n in parent step8; do
  serve $n 8200 || { echo "serve failed $n"; kill $srv; continue; }
  timeout 7200 peval/bin/python pe_groups.py http://localhost:8200/v1 variants/$n results/pg_$n dl_parent/validation.json dl_parent/evaluation.json > results/pg_$n.log 2>&1
  kill $srv; wait $srv 2>/dev/null
done
mkdir -p results/pg_parent/reports; echo EVAL2-DONE
