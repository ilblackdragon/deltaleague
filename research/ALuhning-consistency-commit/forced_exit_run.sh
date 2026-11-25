#!/bin/bash
# Forced-exit test with the GPU to itself: pause rfp-pipeline, let in-flight Command A+ requests finish, unload,
# run forced_exit.py, and ALWAYS resume the pipeline (trap on any exit).
set -u
cd "$HOME/deltaleague-exp"
LOG=runs/forced-exit-run.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
resume() {
  systemctl --user kill -s SIGCONT rfp-pipeline && (cd "$HOME/rfp" && sh mark_interruption.sh end)
  log "rfp-pipeline resumed (exit path)"
}
trap resume EXIT
trap 'log "terminated"; exit 1' TERM INT
export PYTHONPATH="$HOME/deltaleague-exp/pylib-fla:$HOME/deltaleague-exp/pylib:$HOME/deltaleague-exp"
(cd "$HOME/rfp" && sh mark_interruption.sh start "Delta League forced-exit test, GPU to itself (Claude, approved by Aaron)")
systemctl --user kill -s SIGSTOP rfp-pipeline
log "rfp-pipeline paused; waiting for in-flight Command A+ requests"
for i in $(seq 1 240); do
  busy=$(curl -s 127.0.0.1:10003/slots | python3 -c "import json,sys;print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo 0)
  [ "$busy" = "0" ] && break
  sleep 15
done
curl -sf 127.0.0.1:8080/unload >/dev/null && log "llama-swap models unloaded"
sleep 20
log "free memory: $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"
"$HOME/.venvs/train/bin/python" forced_exit.py model-orig runs/failure-orig.jsonl data/heldout.jsonl runs/forced-exit.jsonl \
  --batch 8 > runs/forced-exit.log 2>&1 && log "forced-exit test complete" || log "forced-exit test failed"
