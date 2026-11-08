# Delta League proxy calibration on Kaggle (GPU T4 x2, Internet ON).
#
# Builds a 500-item public proxy of the private test (thinking ON, the private
# sampler, 8192-token cap), runs several leaderboard checkpoints on it (one model
# per GPU, two at a time) and prints proxy accuracy/truncation next to the
# official private-test score. Paste into ONE cell. Results land in
# /kaggle/working/calib/ (summary.json + per-model item records).
#
# Only public datasets (test/validation splits) are used for evaluation; nothing
# here touches the private test.

import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working/calib"); WORK.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("/tmp/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

# version_id -> (label, official private test score on public-ladder500-8k-128-v1)
MODELS = {
    "0ea62f40-7cd8-4d98-9ebb-484192fa05eb": ("original", 17.4),
    "a9340e14-190b-473a-952f-65722b37d576": ("3-in-1", 27.6),
    "b8af1dbc-d52b-4bbc-a302-3505d897303c": ("little-qwen", 32.8),
    "344c8376-66a1-413b-abca-da2b5b986044": ("rep-edit", 35.0),
    "cd0b95bb-c5c0-4656-b9be-ad6482f04075": ("leader", 37.2),
}
N_PER_SOURCE = 100
SEED = 1234


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = r.stdout + r.stderr
    print(out[-2000:], flush=True)
    return r.returncode, out


# ---------- 1. dependencies ----------
try:
    import vllm  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q -U vllm")
sh(f"{sys.executable} -m pip install -q math-verify datasets requests")

# ---------- 2. checkpoints from the public API ----------
import requests

API = "https://deltaleague.ai/api/v1"


def pull(version_id):
    out = MODELS_DIR / version_id[:8]
    out.mkdir(exist_ok=True)
    meta = requests.get(f"{API}/model-versions/{version_id}", timeout=60).json()
    for f in meta["files"]:
        p = out / f["name"]
        if p.exists() and p.stat().st_size == f["size_bytes"]:
            continue
        h = hashlib.sha256()
        with requests.get(f["url"] if "url" in f else f["download_url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(p, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk); h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"hash mismatch {p}"
    print("pulled", version_id[:8], meta["name"][:60], flush=True)
    return out


# ---------- 3. proxy item set ----------
ITEMS = WORK / "items.json"
if not ITEMS.exists():
    import random
    from datasets import load_dataset

    rng = random.Random(SEED)
    items = []
    L = "ABCDEFGHIJ"

    def pick(rows, n):
        rows = list(rows); rng.shuffle(rows); return rows[:n]

    for r in pick(load_dataset("openai/gsm8k", "main", split="test"), N_PER_SOURCE):
        items.append({"src": "gsm8k", "kind": "math", "q": r["question"],
                      "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    for r in pick(load_dataset("HuggingFaceH4/MATH-500", split="test"), N_PER_SOURCE):
        items.append({"src": "math500", "kind": "math", "q": r["problem"], "gold": r["answer"]})
    for r in pick(load_dataset("TIGER-Lab/MMLU-Pro", split="test"), N_PER_SOURCE):
        opts = "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"]))
        items.append({"src": "mmlu_pro", "kind": "mc", "q": f"{r['question']}\n\n{opts}", "gold": r["answer"]})
    for r in pick(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), N_PER_SOURCE):
        labels = r["choices"]["label"]
        opts = "\n".join(f"{L[i]}. {t}" for i, t in enumerate(r["choices"]["text"]))
        items.append({"src": "arc_challenge", "kind": "mc", "q": f"{r['question']}\n\n{opts}",
                      "gold": L[labels.index(r["answerKey"])]})
    for r in pick(load_dataset("tau/commonsense_qa", split="validation"), N_PER_SOURCE):
        opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        items.append({"src": "commonsense_qa", "kind": "mc", "q": f"{r['question']}\n\n{opts}", "gold": r["answerKey"]})
    for i, it in enumerate(items):
        it["id"] = i
    ITEMS.write_text(json.dumps(items))
    print("items:", len(items), flush=True)

# ---------- 4. per-model worker (runs in its own process, one GPU) ----------
WORKER = r'''
import gzip, json, re, sys, time
from vllm import LLM, SamplingParams
from math_verify import parse, verify

def boxed(text):
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0: return out
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0); k += 1
        out.append(text[j + 7:k - 1]); i = k

def score(it, text, finished):
    if not finished or "</think>" not in text: return 0
    ans = text.split("</think>")[-1]
    b = boxed(ans)
    if it["kind"] == "mc":
        pred = b[-1] if b else (re.findall(r"answer is \(?([A-J])\)?", ans) or [""])[-1]
        pred = re.sub(r"[^A-J]", "", pred.replace("\\text", "").upper())[:1]
        return int(pred == it["gold"])
    if not b: return 0
    if it["src"] == "gsm8k":
        p = b[-1].replace(",", "").replace("$", "").replace("\\%", "").strip()
        try: return int(abs(float(p) - float(it["gold"])) < 1e-6)
        except ValueError: pass
    try: return int(verify(parse("$" + it["gold"] + "$"), parse("$" + b[-1] + "$")))
    except Exception: return 0

def main():
    model_dir, items_path, out_path = sys.argv[1:4]
    items = json.load(open(items_path))
    llm = LLM(model_dir, dtype="float16", max_model_len=10240, gpu_memory_utilization=0.88,
              max_num_seqs=128, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    suffix = {"math": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
              "mc": "\n\nAnswer with the letter of the correct option inside \\boxed{}."}
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + suffix[it["kind"]]}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in items]
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=8192, seed=0)
    t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t
    recs = []
    for it, o in zip(items, outs):
        c = o.outputs[0]
        fin = c.finish_reason != "length"
        recs.append({"id": it["id"], "src": it["src"], "finished": fin,
                     "len": len(c.token_ids), "correct": score(it, c.text, fin), "text": c.text})
    with gzip.open(out_path, "wt") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    n = len(recs)
    print(f"DONE {model_dir}: acc {100*sum(r['correct'] for r in recs)/n:.1f} "
          f"trunc {100*sum(not r['finished'] for r in recs)/n:.1f} tokens {sum(r['len'] for r in recs)} "
          f"secs {dt:.0f}", flush=True)

if __name__ == "__main__":
    main()
'''
(WORK / "worker.py").write_text(WORKER)

# ---------- 5. schedule: two models at a time, one per GPU ----------
queue = list(MODELS)  # original first: it is the slowest
running = {}
paths = {}
t0 = time.time()
while queue or running:
    for gpu in ("0", "1"):
        if gpu not in running and queue:
            vid = queue.pop(0)
            out = WORK / f"{vid[:8]}.jsonl.gz"
            if out.exists():
                print("skip (done)", vid[:8]); continue
            mdir = pull(vid)
            log = open(WORK / f"{vid[:8]}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            running[gpu] = (vid, subprocess.Popen(
                [sys.executable, str(WORK / "worker.py"), str(mdir), str(ITEMS), str(out)],
                stdout=log, stderr=subprocess.STDOUT, env=env))
            print(f"[{time.time()-t0:5.0f}s] start {MODELS[vid][0]} on GPU{gpu}", flush=True)
    for gpu, (vid, p) in list(running.items()):
        if p.poll() is not None:
            print(f"[{time.time()-t0:5.0f}s] {MODELS[vid][0]} exit {p.returncode}", flush=True)
            if p.returncode:
                print(open(WORK / f"{vid[:8]}.log").read()[-3000:])
            del running[gpu]
    time.sleep(20)

# ---------- 6. summary ----------
import gzip

rows = []
for vid, (label, official) in MODELS.items():
    f = WORK / f"{vid[:8]}.jsonl.gz"
    if not f.exists():
        continue
    recs = [json.loads(l) for l in gzip.open(f, "rt")]
    n = len(recs)
    by_src = {}
    for r in recs:
        by_src.setdefault(r["src"], []).append(r["correct"])
    rows.append({"version": vid[:8], "label": label, "official_test": official,
                 "proxy_acc": round(100 * sum(r["correct"] for r in recs) / n, 1),
                 "proxy_trunc": round(100 * sum(not r["finished"] for r in recs) / n, 1),
                 "tokens": sum(r["len"] for r in recs),
                 "by_src": {k: round(100 * sum(v) / len(v), 1) for k, v in by_src.items()}})
(WORK / "summary.json").write_text(json.dumps(rows, indent=1))
print("\nSUMMARY")
for r in rows:
    print(json.dumps({k: v for k, v in r.items()}))
if len(rows) >= 3:
    import numpy as np
    a = np.array([r["official_test"] for r in rows]); b = np.array([r["proxy_acc"] for r in rows])
    rank = lambda x: np.argsort(np.argsort(x))
    print("pearson", round(float(np.corrcoef(a, b)[0, 1]), 3),
          "spearman", round(float(np.corrcoef(rank(a), rank(b))[0, 1]), 3))
