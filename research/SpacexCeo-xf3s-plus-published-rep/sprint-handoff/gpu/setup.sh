set -e
. $HOME/venv/bin/activate
mkdir -p $HOME/dl/versions $HOME/dl/results && cd $HOME/dl
python -c "import vllm,transformers;print('vllm',vllm.__version__,'tf',transformers.__version__)"
for v in ab592312-ec77-4f98-b17f-6ea17723b74e 0ea62f40-7cd8-4d98-9ebb-484192fa05eb; do
  [ -f versions/$v/model/model.safetensors ] || delta competition init qwen35-08b-v1 --parent $v --output versions/$v > /dev/null
done
python build_proxy.py 120 80 80 40 proxy400.json
echo SETUP_OK
