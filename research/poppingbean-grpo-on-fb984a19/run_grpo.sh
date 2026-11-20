#!/bin/bash
# GRPO continuation of fb984a19 (rank 1, private 53.4) on a rented NVIDIA GPU (Linux, CUDA >= 12.8).
# Recipe: roshaans' grpo.py from ilblackdragon/deltaleague#58 (evaluation-shaped mix, 6k training cap,
# sampler-matched log-probs, strict reward), unchanged. The only new variable is the parent.
# Upload this folder to ~/work, run `delta login` once yourself, then: bash run_grpo.sh
# Hard time cap: TRAIN_MIN (default 240) for training, then local probes of the checkpoints.
set -x
cd ~/work
export VLLM_USE_FLASHINFER_SAMPLER=0
TRAIN_MIN=${TRAIN_MIN:-420}   # RTX 6000 Ada 48 GB: about 2x slower than A100
JOB_END=$(( $(date +%s) + ${JOB_HOURS:-11}*3600 ))   # whole-job cap, about $11 at $0.99/h
mkdir -p results variants
[ -d deltaleague ] || tar xzf deltaleague-main.tgz
cp src/*.py src/*.sh .
command -v uv >/dev/null || pip install -q uv
[ -x venv/bin/python ] || { uv venv -q --python 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install -q vllm "./deltaleague[local,benchmarks]" flash-linear-attention; }
PY=$PWD/venv/bin/python; D=$PWD/venv/bin/delta
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $*"; sleep 30; done; return 1; }

# base model files (processor configs) + parent weights
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
[ -s variants/parent/model.safetensors ] || { echo SETUP-FAILED; exit 1; }
echo setup-done

# GRPO: 24 steps, checkpoints at 8/16/24, stop early if training reward falls
VLLM_ALLOW_INSECURE_SERIALIZATION=1 GPU_MEM=${GPU_MEM:-0.35} timeout $((TRAIN_MIN*60)) $PY grpo.py variants/parent run1 \
  --steps 24 --save-at 8,16,24 --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 \
  --train-max-tokens 6144 --match-sampler --stop-on-decline 0.05 > results/train.log 2>&1
cp run1/train_log.jsonl results/ 2>/dev/null

# local checks: (1) roshaans' held-out math probe, (2) the official public suite in the thinking-8k config
# served by vLLM. Same items and seeds for the parent and every checkpoint.
evalone() {
  n=$1; port=$2
  GPU_MEM=0.45 $PY probe.py variants/$n results/hold_$n.jsonl --n 256 --seed 7 > results/hold_$n.log 2>&1
  $PY -m vllm.entrypoints.openai.api_server --model variants/$n --served-model-name candidate --port $port       --dtype bfloat16 --max-model-len 34816 --gpu-memory-utilization 0.45 --seed 0 > results/serve_$n.log 2>&1 &
  srv=$!
  for _ in $(seq 120); do curl -sf localhost:$port/v1/models >/dev/null && break; sleep 5; done
  h=$(sha256sum variants/$n/model.safetensors | cut -c1-64)
  timeout 7200 $D public-eval run --endpoint http://localhost:$port/v1 --tokenizer variants/$n --content-hash $h       --output results/pub_$n --manifest dl_parent/validation.json --evaluation-config dl_parent/evaluation.json       > results/pub_$n.log 2>&1
  kill $srv; wait $srv 2>/dev/null
}
for n in parent $(ls -d run1/step* 2>/dev/null | xargs -n1 basename | grep -v "^step8$" | sort -r); do
  [ $(date +%s) -lt $JOB_END ] || { echo "job cap reached, skipping $n"; continue; }
  [ $n = parent ] || { mkdir -p variants/$n; cp run1/$n/* variants/$n/; }
  evalone $n 8100
done
grep -h validation_score results/pub_*.log
echo ALLDONE
