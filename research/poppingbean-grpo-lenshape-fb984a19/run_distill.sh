#!/bin/bash
# 9B-teacher distillation onto fb984a19 on one RTX 6000 Ada (48 GB, needs ~100 GB container disk).
# 1) Qwen3.5-9B answers ~4,000 public train prompts twice (eval sampler, thinking on, 6k cap)
# 2) keep the shortest correct trace per prompt (<= 4,096 tokens)  3) full SFT of fb984a19, 1 epoch
# 4) candidates: sft and blend50 (parent + 0.5 * delta)  5) probe + public groups for candidates and parent
# Whole job capped at JOB_HOURS (default 9).  Run after `delta login`:  bash run_distill.sh
set -x
cd ~/work
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$PWD/venv/bin:$PATH
PY=$PWD/venv/bin/python; D=$PWD/venv/bin/delta
JOB_END=$(( $(date +%s) + ${JOB_HOURS:-9}*3600 ))
left() { echo $(( (JOB_END - $(date +%s)) / 60 )); }
mkdir -p results variants
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $*"; sleep 30; done; return 1; }

# parent model dir (config/tokenizer files from the base repo, weights from fb984a19)
if [ ! -s variants/parent/model.safetensors ]; then
  retry $D competition init qwen35-08b-v1 --output experiment
  $PY -c "
from huggingface_hub import hf_hub_download, list_repo_files
for f in list_repo_files('Qwen/Qwen3.5-0.8B'):
    if 'preprocessor' in f or f == 'processor_config.json':
        hf_hub_download('Qwen/Qwen3.5-0.8B', f, local_dir='experiment/model')
"
  retry $D competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output dl_parent > /dev/null 2>&1
  mkdir -p variants/parent
  cp experiment/model/*.json experiment/model/*.jinja experiment/model/merges.txt variants/parent/
  cp dl_parent/model/model.safetensors variants/parent/
  rm -f experiment/model/model.safetensors
fi
[ -s variants/parent/model.safetensors ] || { echo SETUP-FAILED; exit 1; }
echo setup-done

# 1) teacher traces (cap: 4 h or the job budget minus 4 h for training + evaluation)
TMIN=$(( $(left) - 240 )); [ $TMIN -gt 240 ] && TMIN=240
$PY gen_teacher.py results/teacher.jsonl --prompts 4000 --n 2 --max-tokens 6144 --deadline-minutes $TMIN > results/teacher.log 2>&1
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B   # free 19 GB once the teacher is done

# 2) selection  3) SFT + blend
$PY select_traces.py results/teacher.jsonl results/train.jsonl --max-tokens 4096 > results/select.log 2>&1
$PY train_sft.py variants/parent results/train.jsonl variants --lr 5e-6 --epochs 1 --blend 0.5 > results/sft.log 2>&1

# 5) evaluation: probe + public groups (same driver as the GRPO run), candidates first, parent last
serve() { $PY -m vllm.entrypoints.openai.api_server --model variants/$1 --served-model-name candidate --port $2 \
  --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0 > results/serve_$1.log 2>&1 & srv=$!
  for _ in $(seq 180); do curl -sf localhost:$2/v1/models >/dev/null && return 0; sleep 5; done; return 1; }
for n in sft blend50 parent; do
  [ -s variants/$n/model.safetensors ] || continue
  [ $(left) -gt 50 ] || { echo "job cap: skipping $n"; continue; }
  GPU_MEM=0.85 $PY probe.py variants/$n results/hold_$n.jsonl --n 256 --seed 7 > results/hold_$n.log 2>&1
  serve $n 8200 || { echo "serve failed $n"; kill $srv; continue; }
  timeout $(( $(left) * 60 )) peval/bin/python pe_groups.py http://localhost:8200/v1 variants/$n results/pg_$n \
    dl_parent/validation.json dl_parent/evaluation.json > results/pg_$n.log 2>&1
  kill $srv; wait $srv 2>/dev/null
done
grep -hE "^all " results/hold_*.log; grep -h MEAN results/pg_*.log
echo ALLDONE
