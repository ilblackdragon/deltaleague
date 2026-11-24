#!/bin/bash
# poll a submission until it leaves verifying/queued/running; print final status JSON summary
cd /home/user/delta && source .venv/bin/activate
for i in $(seq 1 120); do
  s=$(delta submission status $1 2>/dev/null)
  st=$(echo "$s" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'])" 2>/dev/null)
  case "$st" in verified|failed|rejected|error|complete) echo "$s" | python3 -c "
import json,sys; d=json.load(sys.stdin); r=d.get('result') or {}
print('STATE', d['state'], d.get('error_code'), 'version', d.get('version_id'))
print('VAL', r.get('validation_score'), 'TEST', r.get('test_score'), 'stats', (r.get('stats') or {}).get('truncation_rate'))
print(json.dumps(r.get('validation_benchmarks')))"; exit 0;; esac
  sleep 30
done
echo TIMEOUT $st
