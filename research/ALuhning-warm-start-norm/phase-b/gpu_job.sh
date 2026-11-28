#!/usr/bin/env bash
# Run via systemd-run --user with RuntimeMaxSec and a stop timeout.
set -euo pipefail
cd "$HOME/deltaleague-exp"
exec 9>runs/phaseB-gpu.lock
flock -n 9
marked=0
paused=0
command_a=http://127.0.0.1:10003
llama_swap=http://127.0.0.1:8080
mark="$HOME/rfp/mark_interruption.sh"

resume() {
    local status=$?
    trap - EXIT
    set +e
    if (( paused )); then
        if podman container exists dl-r2-vllm; then
            podman stop --time 20 dl-r2-vllm || status=1
        fi
        if podman container exists dl-r2-vllm; then
            echo "vLLM is still present, holding RFP paused to avoid sharing the GPU" >&2
            exit 1
        fi
        systemctl --user kill -s SIGCONT rfp-pipeline || status=1
    fi
    if (( marked )); then
        sh "$mark" end || status=1
    fi
    if (( paused )); then
        group=$(systemctl --user show rfp-pipeline -p ControlGroup --value)
        python3 - "$group" <<'PY' || status=1
import sys
from pathlib import Path

group = Path("/sys/fs/cgroup" + sys.argv[1]) / "cgroup.procs"
pids = group.read_text().splitlines()
if not pids:
    raise SystemExit("RFP pipeline cgroup has no processes")
for pid in pids:
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] in {"T", "t"}:
        raise SystemExit(f"RFP pipeline process {pid} is stopped")
print("RFP pipeline resumed", flush=True)
PY
    fi
    exit "$status"
}

slots_busy() {
    curl -fsS --max-time 8 "$command_a/slots" | python3 -c '
import json, sys
slots = json.load(sys.stdin)
if not slots or not all(isinstance(s.get("is_processing"), bool) for s in slots):
    sys.exit("Unexpected slots response")
print(sum(s["is_processing"] for s in slots))
'
}

trap resume EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ $(systemctl --user show dl-phaseA2 -p ActiveState --value) == active ]]; then
    echo "Waiting for the existing Phase A unit to exit"
    while [[ $(systemctl --user show dl-phaseA2 -p ActiveState --value) == active ]]; do
        sleep 15
    done
fi
if podman container exists dl-vllm || podman container exists dl-r2-vllm; then
    echo "Existing Delta League GPU job is still active" >&2
    exit 1
fi
systemctl --user is-active --quiet rfp-pipeline
echo "Waiting for Command A+ to become ready and have a busy slot"
busy=0
for ((i=0; i<240; i++)); do
    if curl -fsS --max-time 8 "$command_a/health" >/dev/null &&
        busy=$(slots_busy 2>/dev/null) && (( busy > 0 )); then
        break
    fi
    busy=0
    sleep 5
done
(( busy > 0 ))
sh "$mark" start "Delta League round 2: $1"
marked=1
paused=1
systemctl --user kill -s SIGSTOP rfp-pipeline
echo "RFP pipeline paused. Waiting for in-flight Command A+ requests to drain."
drained=0
for ((i=0; i<240; i++)); do
    if busy=$(slots_busy 2>/dev/null) && (( busy == 0 )); then
        drained=1
        break
    fi
    sleep 15
done
(( drained ))
curl -fsS --max-time 90 "$llama_swap/unload" >/dev/null
curl -fsS --max-time 8 "$llama_swap/running" |
    python3 -c 'import json, sys; assert json.load(sys.stdin)["running"] == []'
sleep 20
echo "llama-swap unloaded. Starting $1."
bash "$@"
