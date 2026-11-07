#!/bin/bash
# usage: grpo_native_loop.sh RUN BASE_DIR ROUNDS PROMPTS SAMPLES STEP_NORM [CTRL]
# GRPO on native-format train prompts (native/native_pool.json), reward = lm-eval metric.
cd "$(dirname "$0")"
export PATH=$PWD/venv/bin:$PATH HF_HOME=$PWD/hf VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
RUN=$1; BASE=$2; ROUNDS=$3; N=$4; S=$5; NORM=$6; CTRL=${7:-4}
D=grpo/$RUN; LOG=$D/progress.log; mkdir -p $D
score() { CUDA_VISIBLE_DEVICES= HF_HOME=$PWD/native/hf native/venv/bin/python native/score_native.py "$@"; }
[ -f $D/master.safetensors ] || python grpo_tools.py init $BASE $D/master.safetensors >> $LOG 2>&1
for r in $(seq 1 $ROUNDS); do
  [ -f $D/stop ] && break
  [ -f $D/round$r.done ] && continue
  python grpo_tools.py export $BASE $D/master.safetensors $D/serve >> $LOG 2>&1
  python grpo_tools.py sample native/native_pool.json $D/prompts_$r.json $N $((9000 + r)) >> $LOG 2>&1
  rm -rf $D/gen_$r
  python native/run_proxy_native.py $D/serve $D/prompts_$r.json $D/gen_$r --samples $S --seed $((6000 + r)) > $D/gen_$r.log 2>&1
  score $D/prompts_$r.json $D/gen_$r/rows.jsonl --out $D/gen_$r/scores.json > $D/score_$r.log 2>&1 || { echo "round $r score failed" >> $LOG; break; }
  python merge_native.py $D/gen_$r/rows.jsonl $D/gen_$r/scores.json $D/gen_$r/official.jsonl >> $LOG 2>&1 || { echo "round $r merge failed" >> $LOG; break; }
  venv-train/bin/python grpo_step.py $D/master.safetensors $D/gen_$r/official.jsonl $D/prompts_$r.json $D/next.safetensors \
    --config-dir $BASE --layers 0-23 --reward official --step-norm $NORM --skip-first --seed $r > $D/step_$r.log 2>&1 || { echo "round $r step failed" >> $LOG; break; }
  mv $D/next.safetensors $D/master.safetensors
  echo "round $r $(grep -m1 groups $D/step_$r.log)" >> $LOG
  touch $D/round$r.done
  if [ $((r % CTRL)) -eq 0 ]; then
    python grpo_tools.py export $BASE $D/master.safetensors $D/ckpt_$r >> $LOG 2>&1
    python native/run_proxy_native.py $D/ckpt_$r native/native_measure.json results/native_${RUN}r${r}_seed1000 --samples 1 --seed 1000 > $D/nat_$r.log 2>&1
    score native/native_measure.json results/native_${RUN}r${r}_seed1000/rows.jsonl --out results/native_${RUN}r${r}_seed1000/native_scores.json > $D/natscore_$r.log 2>&1
    echo "round $r control $(python3 nat_compare.py n105 ${RUN}r${r} 2>&1 | tail -1)" >> $LOG
  fi
done
touch $D/finished
