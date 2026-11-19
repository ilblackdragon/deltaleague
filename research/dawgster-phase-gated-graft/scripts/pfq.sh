#!/usr/bin/env bash
# usage: scripts/pfq.sh "name:seed name:seed ..."  (model dir models/<name>)
cd "$(dirname "$0")/.."
for job in $1; do n=${job%%:*}; s=${job##*:}; tag=pf_${n}_s$s
  [ -f runs/$tag.scores.json ] && continue
  .venv/bin/python scripts/pubfast_gen.py models/$n runs/$tag.jsonl --seed $s > runs/$tag.log 2>&1 || { echo "$tag GEN FAIL"; continue; }
  DELTA_WASI_ROOT=$PWD/pe/wasi .venv-pe/bin/python scripts/pubfast_score.py runs/$tag.jsonl > /dev/null 2>runs/$tag.score.log || echo "$tag SCORE FAIL"
  python3 -c "import json;s=json.load(open('runs/$tag.scores.json'))['summary'];print('$tag',{k:(v['score'] if isinstance(v,dict) else v) for k,v in s.items()})"
done
