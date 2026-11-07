#!/bin/bash
set -x
cd /root
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p results variants
tar xzf dl-src.tgz
uv venv -q --python 3.12 /root/venv
VIRTUAL_ENV=/root/venv uv pip install -q vllm "./deltaleague[local,benchmarks]" flash-linear-attention
PY=/root/venv/bin/python
D=/root/venv/bin/delta
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "retry $i: $*"; sleep 30; done; return 1; }
retry $D competition init qwen35-08b-v1 --output experiment
$PY -c "
from huggingface_hub import hf_hub_download, list_repo_files
for f in list_repo_files('Qwen/Qwen3.5-0.8B'):
    if 'preprocessor' in f or f == 'processor_config.json':
        hf_hub_download('Qwen/Qwen3.5-0.8B', f, local_dir='experiment/model')
"
retry $D competition init qwen35-08b-v1 --parent 9492e219-6f07-4bdc-b863-e86274ee62c2 --output dl_grpo > /dev/null 2>&1
for v in grpo; do
  mkdir -p variants/$v
  cp experiment/model/*.json experiment/model/*.jinja experiment/model/merges.txt variants/$v/
done
cp dl_grpo/model/model.safetensors variants/grpo/
for f in variants/grpo/config.json variants/grpo/merges.txt variants/grpo/model.safetensors; do
  [ -s "$f" ] || { echo "SETUP-FAILED missing $f"; echo ALLDONE; exit 1; }
done
for v in e14s24 msum mavg; do
  mkdir -p variants/$v
  cp experiment/model/*.json experiment/model/*.jinja experiment/model/merges.txt variants/$v/
done
mv e14s24.safetensors variants/e14s24/model.safetensors
CUDA_VISIBLE_DEVICES= $PY merge.py variants/grpo/model.safetensors variants/msum/model.safetensors variants/e14s24/model.safetensors=1.0 e9s20.safetensors=1.0 > results/build_merge.log 2>&1
CUDA_VISIBLE_DEVICES= $PY merge.py variants/grpo/model.safetensors variants/mavg/model.safetensors variants/e14s24/model.safetensors=0.5 e9s20.safetensors=0.5 >> results/build_merge.log 2>&1
for v in e14s24 msum mavg; do
  [ -s variants/$v/model.safetensors ] || { echo "SETUP-FAILED missing $v"; echo ALLDONE; exit 1; }
done
echo setup-done
printf "hold grpo\ngenx grpo\nhold msum\ngenx msum\nhold mavg\ngenx mavg\n" > jobs_a.txt
P=4 GPU_MEM=0.22 ./run_jobs2.sh jobs_a.txt
echo baselines-done
VLLM_ALLOW_INSECURE_SERIALIZATION=1 GPU_MEM=0.35 timeout 3600 $PY grpo.py variants/e14s24 e15 --steps 12 --save-at 6,12 --mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10 --train-max-tokens 6144 --match-sampler > results/e15_train.log 2>&1
cp e15/train_log.jsonl results/e15_train_log.jsonl
: > jobs_b.txt
for s in 6 12; do
  if [ -f e15/step$s/model.safetensors ]; then
    mkdir -p variants/e15s$s
    cp e15/step$s/* variants/e15s$s/
    printf "hold e15s%s\ngenx e15s%s\n" $s $s >> jobs_b.txt
  fi
done
[ -s jobs_b.txt ] && P=4 GPU_MEM=0.22 ./run_jobs2.sh jobs_b.txt
echo ALLDONE
