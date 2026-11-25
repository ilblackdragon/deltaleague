#!/bin/bash
# Pinned public suite (1,110 items) for the commit LoRA candidate and the original, then delta compare.
# GPU to itself; ALWAYS resumes rfp-pipeline.
set -u
cd "$HOME/deltaleague-exp"
LOG=runs/suite-run.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
resume() {
  podman rm -f dl-vllm >/dev/null 2>&1
  systemctl --user kill -s SIGCONT rfp-pipeline && (cd "$HOME/rfp" && sh mark_interruption.sh end)
  log "rfp-pipeline resumed (exit path)"
}
trap resume EXIT
trap 'log "terminated"; exit 1' TERM INT
export PYTHONPATH="$HOME/deltaleague-exp/pylib-fla:$HOME/deltaleague-exp/pylib:$HOME/deltaleague-exp"
export DELTA_WASI_ROOT="$HOME/deltaleague-exp/.wasi"
PY="$HOME/.venvs/train/bin/python"; PUB="$HOME/deltaleague-exp/.venv-public/bin/delta"; URL=http://127.0.0.1:18000/v1
vllm() { "$PY" -c "import sys; sys.argv=['x']; import grpo_gates as G; print(round(G.start_vllm('$1', 18000, 0.5, $2)))"; }
[ "$(sha256sum validation.json | cut -d' ' -f1)" = "$(python3 -c "import json;print(json.load(open('evaluation.json'))['validation_sha256'])")" ] \
  || { log "validation.json does not match evaluation.json"; exit 1; }

(cd "$HOME/rfp" && sh mark_interruption.sh start "Delta League public suite, original vs commit LoRA (Claude, approved by Aaron)")
systemctl --user kill -s SIGSTOP rfp-pipeline
log "rfp-pipeline paused; waiting for in-flight Command A+ requests"
for i in $(seq 1 240); do
  busy=$(curl -s 127.0.0.1:10003/slots | python3 -c "import json,sys;print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo 0)
  [ "$busy" = "0" ] && break; sleep 15
done
curl -sf 127.0.0.1:8080/unload >/dev/null && log "llama-swap models unloaded"
sleep 20; log "free memory $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"

for arm in lora1 orig; do
  dir=model-$arm; out=runs/public-$arm
  hash=708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8
  [ $arm = lora1 ] && hash=$(sha256sum $dir/model.safetensors | cut -d' ' -f1)   # local label, not the platform hash
  [ -e $out ] && { log "$out exists; refusing to reuse"; exit 1; }
  podman rm -f dl-vllm >/dev/null 2>&1; log "vLLM ($arm) ready in $(vllm $dir 128)s"
  "$PUB" public-eval run --endpoint $URL --model policy --tokenizer "$PWD/model-orig" --content-hash "$hash" \
    --output $out --manifest validation.json --evaluation-config evaluation.json --concurrency 64 >> "$LOG" 2>&1 \
    && log "public suite ($arm) complete" || log "public suite ($arm) failed"
done
"$PUB" compare runs/public-orig/result.json runs/public-lora1/result.json --output runs/public-compare.json >> "$LOG" 2>&1 \
  && log "compare written" || log "compare failed"
log "suite run complete"
