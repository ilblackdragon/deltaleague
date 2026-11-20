#!/bin/bash
# Seed soup of length-shaped GRPO on fb984a19 (#74). c8 (seed 0) is pulled from the platform;
# seeds 1 and 2 are trained here (8 steps each); soups of 2 and 3 checkpoints are evaluated.
# Whole job capped at JOB_HOURS (default 4).  Run after `delta login`:  bash run_soup.sh
set -x
cd ~/work
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/venv/bin:$PATH
PY=$PWD/venv/bin/python; D=$PWD/venv/bin/delta
JOB_END=$(( $(date +%s) + ${JOB_HOURS:-4}*3600 )); left() { echo $(( (JOB_END - $(date +%s)) / 60 )); }
mkdir -p results variants keep
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $*"; sleep 30; done; return 1; }
if [ ! -s variants/parent/model.safetensors ]; then
  retry $D competition init qwen35-08b-v1 --output experiment
  $PY -c "
from huggingface_hub import hf_hub_download, list_repo_files
for f in list_repo_files('Qwen/Qwen3.5-0.8B'):
    if 'preprocessor' in f or f == 'processor_config.json':
        hf_hub_download('Qwen/Qwen3.5-0.8B', f, local_dir='experiment/model')
"
  retry $D competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output dl_parent > /dev/null 2>&1
  mkdir -p variants/parent variants/c8
  cp experiment/model/*.json experiment/model/*.jinja experiment/model/merges.txt variants/parent/
  cp experiment/model/*.json experiment/model/*.jinja experiment/model/merges.txt variants/c8/
  cp dl_parent/model/model.safetensors variants/parent/
  retry $D model pull 4b8a20a6-c78f-4273-adf0-406446c48a55 --output dl_c8
  cp dl_c8/model.safetensors variants/c8/
  rm -f experiment/model/model.safetensors
fi
[ -s variants/c8/model.safetensors ] && [ -s variants/parent/model.safetensors ] || { echo SETUP-FAILED; exit 1; }
echo setup-done
for seed in 1 2; do
  [ -s run_s$seed/step8/model.safetensors ] && continue
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 GPU_MEM=0.35 timeout 2700 $PY grpo.py variants/parent run_s$seed \
    --steps 8 --save-at 8 --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 \
    --train-max-tokens 6144 --match-sampler --len-shape 3000 --seed $seed > results/train_s$seed.log 2>&1
  cp run_s$seed/step8/model.safetensors keep/s${seed}_step8.safetensors 2>/dev/null
done
for n in soup2 soup3; do mkdir -p variants/$n; cp variants/c8/*.json variants/c8/*.jinja variants/c8/merges.txt variants/$n/; done
$PY soup.py variants/soup2/model.safetensors variants/c8/model.safetensors run_s1/step8/model.safetensors
$PY soup.py variants/soup3/model.safetensors variants/c8/model.safetensors run_s1/step8/model.safetensors run_s2/step8/model.safetensors
cp variants/soup2/model.safetensors keep/soup2.safetensors; cp variants/soup3/model.safetensors keep/soup3.safetensors
serve() { $PY -m vllm.entrypoints.openai.api_server --model variants/$1 --served-model-name candidate --port $2 \
  --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0 > results/serve_$1.log 2>&1 & srv=$!
  for _ in $(seq 180); do curl -sf localhost:$2/v1/models >/dev/null && return 0; sleep 5; done; return 1; }
for n in soup3 soup2 c8; do
  [ -s variants/$n/model.safetensors ] || continue
  [ $(left) -gt 45 ] || { echo "job cap: skipping $n"; continue; }
  GPU_MEM=0.85 $PY probe.py variants/$n results/hold_$n.jsonl --n 256 --seed 7 > results/hold_$n.log 2>&1
  serve $n 8400 || { kill $srv; continue; }
  timeout $(( $(left) * 60 )) peval/bin/python pe_groups.py http://localhost:8400/v1 variants/$n results/pg_$n \
    dl_parent/validation.json dl_parent/evaluation.json > results/pg_$n.log 2>&1
  kill $srv; wait $srv 2>/dev/null
done
echo SOUP-DONE
