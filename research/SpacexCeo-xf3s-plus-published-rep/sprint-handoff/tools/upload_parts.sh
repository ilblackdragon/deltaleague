#!/bin/bash
# upload every file of DIR to box (BOX env) as handoff_parts/<name>, retrying a whole part on failure
cd /home/user/delta && source .venv/bin/activate
python tools/gpu.py sh 'mkdir -p $HOME/handoff_parts' 30 >/dev/null
for f in $1/*; do n=$(basename $f)
  for t in 1 2 3 4 5; do python tools/upload_big.py $f handoff_parts/$n > /dev/null 2>&1 && break; sleep 3; done
done
