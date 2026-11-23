#!/bin/bash
cd /root
export VLLM_USE_FLASHINFER_SAMPLER=0
export GPU_MEM=${GPU_MEM:-0.28}
run() {
  kind=$1; v=$2
  case $kind in
    probe) log=results/big_$v.log; cmd=(probe.py variants/$v results/big_$v.jsonl --n 256 --seed 1) ;;
    calib) log=results/calib_$v.log; cmd=(probe.py variants/$v results/calib_$v.jsonl --n 256 --seed 3) ;;
    hold) log=results/hold_$v.log; cmd=(probe.py variants/$v results/hold_$v.jsonl --n 256 --seed 7) ;;
    fresh) log=results/fresh_$v.log; cmd=(probe.py variants/$v results/fresh_$v.jsonl --n 512 --seed 3) ;;
    persona) log=results/fresh_p_$v.log; cmd=(env PERSONA=$v /root/venv/bin/python probe.py variants/lead results/fresh_p_$v.jsonl --n 512 --seed 3) ;;
    gen) log=results/gen_$v.log; cmd=(gen_val.py variants/$v experiment results/gen_$v.jsonl) ;;
    genx) log=results/genx_$v.log; cmd=(env EXTRA=128 /root/venv/bin/python gen_val.py variants/$v experiment results/genx_$v.jsonl) ;;
  esac
  for attempt in $(seq 40); do
    exec 9>/tmp/start.lock
    flock 9
    if [ "${cmd[0]}" = env ]; then "${cmd[@]}" > $log 2>&1 & else /root/venv/bin/python "${cmd[@]}" > $log 2>&1 & fi
    pid=$!
    for _ in $(seq 150); do
      grep -qE "init engine|Processed prompts|Traceback" $log && break
      kill -0 $pid 2>/dev/null || break
      sleep 2
    done
    flock -u 9
    wait $pid
    status=$?
    grep -qE "Error in memory profiling|less than desired GPU memory" $log || break
    sleep 60
  done
  echo "finished $kind $v exit=$status" >> results/jobs.log
}
export -f run
xargs -P ${P:-3} -L 1 bash -c 'run "$@"' _ < "$1"
echo "alldone $1" >> results/jobs.log
