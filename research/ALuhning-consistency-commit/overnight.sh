#!/bin/bash
# Overnight Delta League run on aitterminator (original-model lineage only).
#   1. pause rfp-pipeline (SIGSTOP + interruption mark), let in-flight Command A+ requests finish, unload
#   2. filter: original model, 1,024 gsm8k TRAIN problems x 2 samples (cap 6,144) -> problems it can solve
#   3. gate-only GRPO on that pool (6 steps, 12 prompts x 16, cap 8,192; held-out evals at 3 and 6; kill rule)
#   4. pinned public suite (1,110 items) for the original and the trained checkpoint, paired compare
#   5. ALWAYS: stop vLLM, resume rfp-pipeline, close the interruption mark (trap on any exit)
# Run as: systemd-run --user --unit=dl-overnight -p RuntimeMaxSec=12h bash overnight.sh
set -u
cd "$HOME/deltaleague-exp"
mkdir -p runs
LOG=runs/overnight.log
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
PY="$HOME/.venvs/train/bin/python"
PUB="$HOME/deltaleague-exp/.venv-public/bin/delta"
vllm() { "$PY" -c "import sys; sys.argv=['x']; import grpo_gates as G; print(round(G.start_vllm('$1', 18000, 0.5, $2)))"; }

# 1. pause
(cd "$HOME/rfp" && sh mark_interruption.sh start "Delta League overnight: filter + gate-only GRPO + public suite (Claude, approved by Aaron)")
systemctl --user kill -s SIGSTOP rfp-pipeline
log "rfp-pipeline paused; waiting for in-flight Command A+ requests"
for i in $(seq 1 240); do
  busy=$(curl -s 127.0.0.1:10003/slots | python3 -c "import json,sys;print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo 0)
  [ "$busy" = "0" ] && break
  sleep 15
done
curl -sf 127.0.0.1:8080/unload >/dev/null && log "llama-swap models unloaded"
sleep 20

# 2. filter
python3 -c "
import json
rows=[json.loads(l) for l in open('data/train.jsonl')]
g=[r for r in rows if r['bench']=='gsm8k'][:1024]
open('data/filter-candidates.jsonl','w').write(''.join(json.dumps(r)+'\n' for r in g))
print(len(g))" | xargs -I{} sh -c 'echo "$(date "+%F %T") filter candidates: {}" >> runs/overnight.log'
log "vLLM (original) ready in $(vllm model-orig 256)s"
"$PY" think_probe.py data/filter-candidates.jsonl runs/filter.jsonl policy=http://127.0.0.1:18000/v1 \
  --samples 2 --cap 6144 --gen evaluation.json --conc 256 --save-text --seed 20261125 >> "$LOG" 2>&1 || { log "filter failed"; exit 1; }
"$PY" make_pool.py runs/filter.jsonl data/filter-candidates.jsonl data/pool.jsonl >> "$LOG" 2>&1
podman rm -f dl-vllm >/dev/null 2>&1
n=$(wc -l < data/pool.jsonl); log "pool size $n"
[ "$n" -ge 12 ] || { log "pool too small to train; stopping"; exit 1; }

# 3. gate-only GRPO
"$PY" grpo_gates.py --parent model-orig --data data --pool data/pool.jsonl --out runs/gates3 --steps 6 \
  --prompts 12 --group 16 --cap 8192 --lr 3e-4 --eval-every 3 --kill-truncation 0.8875 \
  --max-seqs 256 --util 0.5 --conc 256 --max-neg 4 --gen evaluation.json >> "$LOG" 2>&1 || log "trainer exited non-zero"
grep -q "KILL RULE" runs/gates3/log.txt && { log "kill rule fired; skipping public suite"; exit 0; }
[ -d runs/gates3/policy-step6 ] || { log "no final policy; skipping public suite"; exit 1; }

# 4. public suite, original vs trained (thinking off; coding/BFCL greedy 4K; pinned manifest)
for arm in orig trained; do
  dir=model-orig; [ $arm = trained ] && dir=runs/gates3/policy-step6
  hash=708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8
  [ $arm = trained ] && hash=$(sha256sum $dir/model.safetensors | cut -d' ' -f1)   # local label, not the platform hash
  log "vLLM ($arm) ready in $(vllm $dir 64)s"
  "$PUB" public-eval run --endpoint http://127.0.0.1:18000/v1 --model policy --tokenizer "$PWD/model-orig" \
    --content-hash "$hash" --output runs/public-$arm --evaluation-config evaluation.json --concurrency 64 >> "$LOG" 2>&1 \
    || log "public suite ($arm) failed"
  podman rm -f dl-vllm >/dev/null 2>&1
done
"$PUB" compare runs/public-orig/result.json runs/public-trained/result.json --output runs/public-compare.json >> "$LOG" 2>&1 \
  || log "compare failed"
log "overnight run complete"
