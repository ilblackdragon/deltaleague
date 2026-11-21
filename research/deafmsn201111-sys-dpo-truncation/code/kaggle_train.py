# Step 3 on Kaggle (GPU T4 x2, Internet ON): train the LoRA on the leader, merge it, and evaluate on the proxy.
#
# Inputs: a Kaggle Dataset with the contents of ~/Downloads/deltaleague-exp/kaggle_upload/
#   (sft.jsonl, proxy_items.json, lora_train.py, merge_lora.py, eval_worker.py), added via "Add Input".
# GPU0: train LoRA r16 on the leader cd0b95bb -> merge at dose 1.0 and 0.5 -> evaluate dose 1.0.
# GPU1: evaluate the leader in the same session (paired baseline) -> evaluate dose 0.5.
# Output: /kaggle/working/train.tgz (LoRA, eval records, logs, summary). Merged weights are NOT packed
# (1.7 GB); rebuild them locally with merge_lora.py.
#
# Run via Save Version -> Save & Run All (Commit). Expected 3-5 h.

import glob, gzip, hashlib, json, os, random, shutil, subprocess, sys, time
from pathlib import Path

W = Path("/kaggle/working/train"); W.mkdir(parents=True, exist_ok=True)
M = Path("/tmp/models"); M.mkdir(parents=True, exist_ok=True)
LEADER = "cd0b95bb-c5c0-4656-b9be-ad6482f04075"
TRAIN_MINUTES = 240  # hard cap on LoRA training time


def sh(cmd, tail=2000):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print((r.stdout + r.stderr)[-tail:], flush=True)
    return r.returncode


inp = Path(glob.glob("/kaggle/input/**/sft.jsonl", recursive=True)[0]).parent
print("input dir:", inp, flush=True)
for f in ["lora_train.py", "merge_lora.py", "eval_worker.py", "proxy_items.json", "sft.jsonl"]:
    shutil.copy(inp / f, W / f)

try:
    import vllm  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q -U vllm")
sh(f"{sys.executable} -m pip install -q math-verify requests")
sh(f"{sys.executable} -c 'import transformers, torch; print(transformers.__version__, torch.__version__)'")

import requests


def pull(version_id):
    out = M / version_id[:8]
    out.mkdir(exist_ok=True)
    meta = requests.get(f"https://deltaleague.ai/api/v1/model-versions/{version_id}", timeout=60).json()
    for f in meta["files"]:
        p = out / f["name"]
        if p.exists() and p.stat().st_size == f["size_bytes"]:
            continue
        h = hashlib.sha256()
        with requests.get(f["url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(p, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk); h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"hash mismatch {p}"
    return out


leader_dir = pull(LEADER)


def run_bg(args, gpu, log):
    return subprocess.Popen([sys.executable] + args, stdout=open(W / log, "w"), stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), cwd=W)


def wait(p, log):
    p.wait()
    print(f"--- {log} exit {p.returncode}\n" + open(W / log).read()[-1500:], flush=True)
    return p.returncode


def evaluate(model_dir, tag, gpu):
    return run_bg(["eval_worker.py", str(model_dir), "proxy_items.json", f"eval_{tag}.jsonl.gz"], gpu, f"eval_{tag}.log")


# GPU1: same-session leader baseline
p_leader = evaluate(leader_dir, "leader", "1")

# GPU0: try the fast linear-attention kernels; fall back to the reference implementation if they fail on T4
fla_ok = sh(f"{sys.executable} -m pip install -q flash-linear-attention") == 0
if fla_ok:
    rc = wait(run_bg(["lora_train.py", "--model", str(leader_dir), "--data", "sft.jsonl", "--out", "/tmp/lora_smoke",
                      "--limit", "4", "--grad-accum", "2", "--max-len", "2048"], "0", "smoke.log"), "smoke.log")
    if rc != 0:
        print("FLA kernels failed on this GPU -> uninstalling, using the reference implementation", flush=True)
        sh(f"{sys.executable} -m pip uninstall -y flash-linear-attention fla-core")

rc = wait(run_bg(["lora_train.py", "--model", str(leader_dir), "--data", "sft.jsonl", "--out", str(W / "lora"),
                  "--max-minutes", str(TRAIN_MINUTES)], "0", "train.log"), "train.log")
assert rc == 0, "training failed"

for dose in ("1.0", "0.5"):
    sh(f"cd {W} && {sys.executable} merge_lora.py {leader_dir} {W / 'lora'} {M / ('cand_d' + dose.replace('.', ''))} --dose {dose}")

p0 = evaluate(M / "cand_d10", "cand_d10", "0")
wait(p_leader, "eval_leader.log")
p1 = evaluate(M / "cand_d05", "cand_d05", "1")
wait(p0, "eval_cand_d10.log")
wait(p1, "eval_cand_d05.log")

# ---------- summary with paired bootstrap vs the same-session leader ----------
def load(tag):
    f = W / f"eval_{tag}.jsonl.gz"
    return {r["id"]: r for r in map(json.loads, gzip.open(f, "rt"))} if f.exists() else None


base = load("leader")
summary = {}
for tag in ("leader", "cand_d10", "cand_d05"):
    recs = load(tag)
    if not recs:
        continue
    n = len(recs)
    by = {}
    for r in recs.values():
        by.setdefault(r["src"], []).append(r["correct"])
    row = {"acc": round(100 * sum(r["correct"] for r in recs.values()) / n, 1),
           "acc_strict_boxed": round(100 * sum(r["correct_strict"] for r in recs.values()) / n, 1),
           "trunc": round(100 * sum(not r["finished"] for r in recs.values()) / n, 1),
           "trunc_after_think": round(100 * sum(not r["finished"] and "</think>" in r["text"] for r in recs.values()) / n, 1),
           "tokens": sum(r["len"] for r in recs.values()),
           "by_src": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
    if base and tag != "leader":
        ids = sorted(set(recs) & set(base))
        d = [recs[i]["correct"] - base[i]["correct"] for i in ids]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) * 100 for _ in range(2000))
        row["diff_vs_leader"] = round(100 * sum(d) / len(d), 1)
        row["ci95"] = [round(boots[50], 1), round(boots[1949], 1)]
    summary[tag] = row
(W / "summary.json").write_text(json.dumps(summary, indent=1))
print("\nSUMMARY")
for k, v in summary.items():
    print(k, json.dumps(v))

sh("cd /kaggle/working && tar czf train.tgz train/lora train/*.jsonl.gz train/*.log train/summary.json && ls -lh train.tgz")
