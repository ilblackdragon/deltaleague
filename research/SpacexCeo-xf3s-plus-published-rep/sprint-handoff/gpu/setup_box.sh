set -x
cd $HOME
pip install --break-system-packages -q uv > uvpip.log 2>&1 || pip install --user --break-system-packages -q uv >> uvpip.log 2>&1
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH; which uv
export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 $HOME/venv && . $HOME/venv/bin/activate && uv pip install vllm "deltaleague[local]" math-verify "lm-eval[ifeval]" > $HOME/pip.log 2>&1
echo PIPDONE >> $HOME/pip.log
cd $HOME/dl
for v in b8af1dbc-d52b-4bbc-a302-3505d897303c; do delta competition init qwen35-08b-v1 --parent $v --output versions/$v >> init.log 2>&1; cp hfextra/*.json versions/$v/model/; done
echo INITDONE >> init.log
pkill -f runner.py; nohup $HOME/venv/bin/python runner.py > runner.log 2>&1 &
echo RUNNER_STARTED >> init.log
