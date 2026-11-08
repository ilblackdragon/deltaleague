#!/bin/bash
# usage: grpo_mixed_loop.sh RUN BASE_DIR ROUNDS N_NATIVE N_OTHER SAMPLES STEP_NORM [CTRL] [CONTROL_BASE]
# GRPO with half native-format math prompts (lm-eval reward) and half other families with varied
# answer instructions (official scorer reward). IFEval is never in the pool.
cd "$(dirname "$0")"
export PATH=$PWD/venv/bin:$PATH HF_HOME=$PWD/hf VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
RUN=$1; BASE=$2; ROUNDS=$3; NN=$4; NO=$5; S=$6; NORM=$7; CTRL=${8:-4}; CBASE=${9:-d105}
D=grpo/$RUN; LOG=$D/progress.log; mkdir -p $D
score() { CUDA_VISIBLE_DEVICES= HF_HOME=$PWD/native/hf native/venv/bin/python native/score_native.py "$@"; }
[ -f proxy/pool5.json ] || python3 -c "
import json
rows=[r for r in json.load(open('proxy/pool4.json')) if r['kind'] not in ('ifeval','code') and r['bench'] not in ('math_train','gsm_train')]
json.dump(rows,open('proxy/pool5.json','w')); print('pool5',len(rows))" >> $LOG
[ -f $D/master.safetensors ] || python grpo_tools.py init $BASE $D/master.safetensors >> $LOG 2>&1
for r in $(seq 1 $ROUNDS); do
  [ -f $D/stop ] && break
  [ -f $D/round$r.done ] && continue
  python grpo_tools.py export $BASE $D/master.safetensors $D/serve >> $LOG 2>&1
  python grpo_tools.py sample native/native_pool.json $D/pn_$r.json $NN $((9100 + r)) >> $LOG 2>&1
  python grpo_tools.py sample proxy/pool5.json $D/po_$r.json $NO $((9200 + r)) >> $LOG 2>&1
  python3 -c "import json;json.dump(json.load(open('$D/pn_$r.json'))+json.load(open('$D/po_$r.json')),open('$D/prompts_$r.json','w'))"
  rm -rf $D/gen_$r $D/genn_$r $D/geno_$r
  python native/run_proxy_native.py $D/serve $D/pn_$r.json $D/genn_$r --samples $S --seed $((6100 + r)) > $D/genn_$r.log 2>&1
  python native/run_proxy_native.py $D/serve $D/po_$r.json $D/geno_$r --samples $S --seed $((6200 + r)) > $D/geno_$r.log 2>&1
  score $D/pn_$r.json $D/genn_$r/rows.jsonl --out $D/genn_$r/scores.json > $D/score_$r.log 2>&1 || { echo "round $r score failed" >> $LOG; break; }
  python merge_native.py $D/genn_$r/rows.jsonl $D/genn_$r/scores.json $D/rn_$r.jsonl >> $LOG 2>&1 || { echo "round $r merge failed" >> $LOG; break; }
  python add_official.py $D/po_$r.json $D/geno_$r/rows.jsonl $D/ro_$r.jsonl >> $LOG 2>&1 || { echo "round $r official failed" >> $LOG; break; }
  cat $D/rn_$r.jsonl $D/ro_$r.jsonl > $D/official_$r.jsonl
  venv-train/bin/python grpo_step.py $D/master.safetensors $D/official_$r.jsonl $D/prompts_$r.json $D/next.safetensors \
    --config-dir $BASE --layers 0-23 --reward official --step-norm $NORM --skip-first --seed $r > $D/step_$r.log 2>&1 || { echo "round $r step failed" >> $LOG; break; }
  mv $D/next.safetensors $D/master.safetensors
  echo "round $r $(grep -m1 groups $D/step_$r.log)" >> $LOG
  touch $D/round$r.done
  if [ $((r % CTRL)) -eq 0 ]; then
    python grpo_tools.py export $BASE $D/master.safetensors $D/ckpt_$r >> $LOG 2>&1
    python native/run_proxy_native.py $D/ckpt_$r native/native_measure.json results/native_${RUN}r${r}_seed1000 --samples 1 --seed 1000 > $D/nat_$r.log 2>&1
    score native/native_measure.json results/native_${RUN}r${r}_seed1000/rows.jsonl --out results/native_${RUN}r${r}_seed1000/native_scores.json > $D/natscore_$r.log 2>&1
    python run_proxy.py $D/ckpt_$r proxy/side180.json results/side_${RUN}r${r}_seed1000 --samples 3 --seed 1000 > $D/side_$r.log 2>&1
    echo "round $r control $(python3 nat_compare.py $CBASE ${RUN}r${r} 2>&1 | tail -1)" >> $LOG
    echo "round $r side $(python3 -c "
import json
r=[json.loads(l) for l in open('results/side_${RUN}r${r}_seed1000/rows.jsonl')];N=len(r)
print('trunc %.1f multi %.1f'%(100*sum(x['truncated'] for x in r)/N,100*sum(x['text'].count('</think>')>1 for x in r)/N))")" >> $LOG
  fi
done
touch $D/finished
