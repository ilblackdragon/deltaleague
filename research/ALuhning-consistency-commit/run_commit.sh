#!/bin/bash
# Consistency-gated commit + half-hint repair, original lineage. GPU to itself; ALWAYS resumes rfp-pipeline.
#   1 pause pipeline, let in-flight Command A+ requests finish, unload
#   2 vLLM(original): commit examples from the filter's training attempts; half-hint MATH repair examples
#   3 LoRA fine-tune -> Delta `lora` patch -> reconstruct with delta_cli -> delta validate
#   4 held-out check (128 problems, sample-0 seeds); kill rule: correct must reach >= 15/128 (original 8/128, +5 pts)
#   5 public suite (1,110 items) for candidate and original, delta compare
set -u
cd "$HOME/deltaleague-exp"
mkdir -p runs; LOG=runs/commit-run.log
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
PY="$HOME/.venvs/train/bin/python"; PUBPY="$HOME/deltaleague-exp/.venv-public/bin/python"; PUB="$HOME/deltaleague-exp/.venv-public/bin/delta"
URL=http://127.0.0.1:18000/v1
vllm() { "$PY" -c "import sys; sys.argv=['x']; import grpo_gates as G; print(round(G.start_vllm('$1', 18000, 0.5, $2)))"; }

(cd "$HOME/rfp" && sh mark_interruption.sh start "Delta League commit+repair LoRA run (Claude, approved by Aaron)")
systemctl --user kill -s SIGSTOP rfp-pipeline
log "rfp-pipeline paused; waiting for in-flight Command A+ requests"
for i in $(seq 1 240); do
  busy=$(curl -s 127.0.0.1:10003/slots | python3 -c "import json,sys;print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo 0)
  [ "$busy" = "0" ] && break; sleep 15
done
curl -sf 127.0.0.1:8080/unload >/dev/null && log "llama-swap models unloaded"
sleep 20; log "free memory $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"

# 2. data
log "vLLM (original) ready in $(vllm model-orig 256)s"
"$PY" commit_data.py runs/filter.jsonl data/filter-candidates.jsonl data/commit.jsonl $URL --max-traces 1200 --conc 256 >> "$LOG" 2>&1 \
  || { log "commit data failed"; exit 1; }
"$PY" hint_data.py gen data/math-hints.jsonl data/repair.jsonl $URL --conc 256 >> "$LOG" 2>&1 || { log "repair data failed"; exit 1; }
podman rm -f dl-vllm >/dev/null 2>&1
log "data: commit+finished $(wc -l < data/commit.jsonl), repair $(wc -l < data/repair.jsonl)"

# 3. train + export + reconstruct + validate
"$PY" train_lora_commit.py --parent model-orig --data data/commit.jsonl data/repair.jsonl --out runs/lora1 \
  --submission runs/lora1/submission >> "$LOG" 2>&1 || { log "training failed"; exit 1; }
PYTHONPATH="$HOME/deltaleague-exp" "$PUBPY" reconstruct_eval.py model-orig runs/lora1/submission model-lora1 >> "$LOG" 2>&1 || { log "reconstruct failed"; exit 1; }
"$PUB" validate runs/lora1/submission >> "$LOG" 2>&1 && log "patch validated" || log "patch validation FAILED"

# 4. held-out check
log "vLLM (candidate) ready in $(vllm model-lora1 128)s"
"$PY" think_probe.py data/heldout.jsonl runs/lora1-heldout.jsonl policy=$URL --samples 1 --gen evaluation.json --conc 128 --save-text >> "$LOG" 2>&1
n=$("$PY" -c "
import json; from grpo_gates import is_correct
g={r['id']:r for r in map(json.loads, open('data/heldout.jsonl'))}; k=0
for r in map(json.loads, open('runs/lora1-heldout.jsonl')):
    t=r.get('text',''); f=t.split('</think>',1)[1] if '</think>' in t else ''
    k += r['finish_reason']=='stop' and is_correct(r['bench'], f, g[r['id']]['answer'])
print(k)")
log "HELDOUT candidate correct $n/128 (original 8/128; kill below 15)"
[ "$n" -ge 15 ] || { log "KILL RULE: below +5 points; skipping public suite"; exit 0; }

# 5. public suite: candidate (server already up), then original
h=$(sha256sum model-lora1/model.safetensors | cut -d' ' -f1)
podman rm -f dl-vllm >/dev/null 2>&1; log "vLLM (candidate, suite) ready in $(vllm model-lora1 64)s"
"$PUB" public-eval run --endpoint $URL --model policy --tokenizer "$PWD/model-orig" --content-hash "$h" \
  --output runs/public-lora1 --evaluation-config evaluation.json --concurrency 64 >> "$LOG" 2>&1 || log "public suite (candidate) failed"
podman rm -f dl-vllm >/dev/null 2>&1; log "vLLM (original, suite) ready in $(vllm model-orig 64)s"
"$PUB" public-eval run --endpoint $URL --model policy --tokenizer "$PWD/model-orig" \
  --content-hash 708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8 \
  --output runs/public-orig --evaluation-config evaluation.json --concurrency 64 >> "$LOG" 2>&1 || log "public suite (original) failed"
"$PUB" compare runs/public-orig/result.json runs/public-lora1/result.json --output runs/public-compare.json >> "$LOG" 2>&1 || log "compare failed"
log "run complete"
