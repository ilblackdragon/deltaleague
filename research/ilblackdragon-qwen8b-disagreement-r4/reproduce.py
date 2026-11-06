"""Run in an initialized workspace, with a single 7200-second outer deadline."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    here = Path(__file__).resolve().parent
    deadline = time.monotonic() + 7200
    env = dict(os.environ, OMP_NUM_THREADS='16', MKL_NUM_THREADS='16',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_DISABLE_PROGRESS_BARS='1')
    def run(argv):
        subprocess.run(argv, cwd=w, env=env, check=True,
                       timeout=max(1, deadline - time.monotonic()))
    download = """
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
    local_dir='teacher', allow_patterns=['*.safetensors', '*.json', '*.jinja', 'merges.txt', 'vocab.json'], max_workers=4)
"""
    run([sys.executable, '-c', download])
    for stage in ['student', 'teacher', 'edit']:
        run([sys.executable, '-u', str(here/'experiment.py'), '--workspace', str(w), '--stage', stage])
    cli = str(Path(sys.executable).parent/'delta')
    run([cli, 'patch', 'build', '--parent', './model', '--modified', './edited.safetensors', '--output', './submission'])
    run([cli, 'validate', './submission'])


if __name__ == '__main__':
    main()
