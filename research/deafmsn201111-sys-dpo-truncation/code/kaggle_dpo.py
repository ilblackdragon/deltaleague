# Option B, step 2 on Kaggle (GPU T4 x2, Internet ON): preference training against 8192-token truncation.
#
# Input: a Kaggle Dataset with ~/Downloads/deltaleague-exp/kaggle_upload3/.
# Base = leader + LoRA v1 (cand_d10). GPU0 trains at lr 2e-5, GPU1 at lr 5e-5, each then evaluates on the proxy.
# The leader and cand_d10 are not re-evaluated: same weights + seed reproduce identical text, records are stored.
# Output: /kaggle/working/dpo.tgz. Run via Save Version -> Save & Run All (Commit). Expected ~2.5-3.5 h.
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

_gpus = subprocess.run("nvidia-smi -L", shell=True, text=True, capture_output=True).stdout
print(_gpus, flush=True)
assert _gpus.count("GPU ") >= 2, "No GPU T4 x2 in this session: set Settings -> Accelerator -> GPU T4 x2 (check the weekly GPU quota)"

W = Path("/kaggle/working/dpo"); W.mkdir(parents=True, exist_ok=True)
M = Path("/tmp/models"); M.mkdir(parents=True, exist_ok=True)
LEADER = "cd0b95bb-c5c0-4656-b9be-ad6482f04075"
CONFIGS = {"0": ("a", "2e-5"), "1": ("b", "5e-5")}


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print((r.stdout + r.stderr)[-2000:], flush=True)
    return r.returncode


inp = Path(glob.glob("/kaggle/input/**/pairs.jsonl", recursive=True)[0]).parent
print("input dir:", inp, flush=True)
for f in ["dpo_train.py", "merge_lora.py", "eval_worker.py", "proxy_items.json", "pairs.jsonl"]:
    shutil.copy(inp / f, W / f)

try:
    import vllm  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q -U vllm")
sh(f"{sys.executable} -m pip install -q math-verify requests flash-linear-attention")

import requests

out = M / LEADER[:8]; out.mkdir(exist_ok=True)
meta = requests.get(f"https://deltaleague.ai/api/v1/model-versions/{LEADER}", timeout=60).json()
for f in meta["files"]:
    h = hashlib.sha256()
    with requests.get(f["url"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out / f["name"], "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk); h.update(chunk)
    assert h.hexdigest() == f["sha256"], f["name"]
base = M / "cand_d10"
assert sh(f"{sys.executable} {W / 'merge_lora.py'} {out} {inp} {base} --dose 1.0") == 0

# One shell chain per GPU: train -> merge -> evaluate.
procs = {}
for gpu, (tag, lr) in CONFIGS.items():
    chain = (f"{sys.executable} dpo_train.py --model {base} --data pairs.jsonl --out lora_{tag} --lr {lr} --epochs 2 --max-minutes 240 "
             f"&& {sys.executable} merge_lora.py {base} lora_{tag} {M / ('dpo_' + tag)} "
             f"&& {sys.executable} eval_worker.py {M / ('dpo_' + tag)} proxy_items.json eval_{tag}.jsonl.gz")
    procs[tag] = subprocess.Popen(chain, shell=True, cwd=W, stdout=open(W / f"{tag}.log", "w"), stderr=subprocess.STDOUT,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
    print(f"started config {tag} lr={lr} on GPU{gpu}", flush=True)
for tag, p in procs.items():
    p.wait()
    log = open(W / f"{tag}.log").read()
    keep = [l for l in log.splitlines() if l.startswith(("pairs", "step", "saved", "merged", "DONE", "time budget"))]
    print(f"--- {tag} exit {p.returncode}\n" + "\n".join(keep[-14:]), flush=True)
    if p.returncode:
        print(log[-2500:], flush=True)
sh("cd /kaggle/working && tar czf dpo.tgz dpo/lora_* dpo/*.log $(ls dpo/*.jsonl.gz 2>/dev/null) && ls -lh dpo.tgz")
