# Step 1 of the distillation experiment: sample training traces on Kaggle (GPU T4 x2, Internet ON).
#
# GPU0: leader cd0b95bb on 2000 TRAIN-split prompts (math + multiple choice), 2 samples each.
# GPU1: 3-in-1 a9340e14 on the 1000 multiple-choice prompts only, 2 samples each.
# The sampler is the same as the private test's (T=1, top_p .95, top_k 20, presence 1.5, 8192 cap).
# Results are written in chunks, so a timeout keeps what has finished. At the end
# everything is packed into /kaggle/working/gen.tgz.
#
# Run via Save Version -> Save & Run All (Commit). Expected ~4-5 h; hard stop at ~10.5 h.
# Only TRAIN splits are used: no overlap with the proxy (test/validation) or the private test.

import json, os, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working/gen"); WORK.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("/tmp/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEADLINE = time.time() + 10.5 * 3600
SEED = 777

JOBS = {  # gpu -> (version_id, which items)
    "0": ("cd0b95bb-c5c0-4656-b9be-ad6482f04075", "all"),
    "1": ("a9340e14-190b-473a-952f-65722b37d576", "mc"),
}


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print((r.stdout + r.stderr)[-2000:], flush=True)
    return r.returncode


try:
    import vllm  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q -U vllm")
sh(f"{sys.executable} -m pip install -q math-verify datasets requests")

import hashlib
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
        with requests.get(f["url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(p, "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk); h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"hash mismatch {p}"
    print("pulled", version_id[:8], meta["name"][:60], flush=True)
    return out


# ---------- train prompts ----------
ITEMS = WORK / "train_items.json"
if not ITEMS.exists():
    import random
    import re
    from datasets import load_dataset

    rng = random.Random(SEED)
    L = "ABCDEFGHIJ"
    items = []

    def pick(rows, n):
        rows = list(rows); rng.shuffle(rows); return rows[:n]

    def last_boxed(s):
        j = s.rfind("\\boxed{")
        if j < 0:
            return None
        k, depth = j + 7, 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        return s[j + 7:k - 1]

    for r in pick(load_dataset("openai/gsm8k", "main", split="train"), 500):
        items.append({"src": "gsm8k", "kind": "math", "q": r["question"],
                      "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    n = 0
    for r in pick(load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train"), 800):
        g = last_boxed(r["solution"])
        if g and n < 500:
            items.append({"src": "math", "kind": "math", "q": r["problem"], "gold": g, "level": r["level"]})
            n += 1
    for r in pick(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"), 300):
        labels = r["choices"]["label"]
        opts = "\n".join(f"{L[i]}. {t}" for i, t in enumerate(r["choices"]["text"]))
        items.append({"src": "arc_challenge", "kind": "mc", "q": f"{r['question']}\n\n{opts}",
                      "gold": L[labels.index(r["answerKey"])]})
    for r in pick(load_dataset("tau/commonsense_qa", split="train"), 400):
        opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        items.append({"src": "commonsense_qa", "kind": "mc", "q": f"{r['question']}\n\n{opts}", "gold": r["answerKey"]})
    for r in pick(load_dataset("cais/mmlu", "auxiliary_train", split="train"), 300):
        r = r.get("train", r)
        opts = "\n".join(f"{L[i]}. {t}" for i, t in enumerate(r["choices"]))
        items.append({"src": "mmlu_aux", "kind": "mc", "q": f"{r['question']}\n\n{opts}", "gold": L[r["answer"]]})
    for i, it in enumerate(items):
        it["id"] = f"tr{i}"
    ITEMS.write_text(json.dumps(items))
    from collections import Counter
    print("train items:", len(items), Counter(it["src"] for it in items), flush=True)

# ---------- worker ----------
WORKER = r'''
import gzip, json, os, re, sys, time
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
        if not b: return 0
        m = re.findall(r"[A-J]", b[-1].replace("\\text", ""))
        return int(bool(m) and m[0] == it["gold"])
    if not b: return 0
    if it["src"] == "gsm8k":
        p = b[-1].replace(",", "").replace("$", "").replace("\\%", "").strip()
        try: return int(abs(float(p) - float(it["gold"])) < 1e-6)
        except ValueError: pass
    try: return int(verify(parse("$" + it["gold"] + "$"), parse("$" + b[-1] + "$")))
    except Exception: return 0

def main():
    model_dir, items_path, out_path, which, deadline = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
    items = json.load(open(items_path))
    if which == "mc":
        items = [it for it in items if it["kind"] == "mc"]
    done = set()
    if os.path.exists(out_path):
        for l in gzip.open(out_path, "rt"):
            done.add(json.loads(l)["id"])
    items = [it for it in items if it["id"] not in done]
    llm = LLM(model_dir, dtype="float16", max_model_len=10240, gpu_memory_utilization=0.88,
              max_num_seqs=128, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    suffix = {"math": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
              "mc": "\n\nAnswer with the letter of the correct option inside \\boxed{}."}
    sp = SamplingParams(n=2, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=8192, seed=0)
    CH = 200
    for s in range(0, len(items), CH):
        if time.time() > deadline:
            print("DEADLINE reached, stopping", flush=True); break
        chunk = items[s:s + CH]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + suffix[it["kind"]]}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in chunk]
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        with gzip.open(out_path, "at") as f:
            for it, o in zip(chunk, outs):
                for k, c in enumerate(o.outputs):
                    fin = c.finish_reason != "length"
                    f.write(json.dumps({"id": it["id"], "src": it["src"], "sample": k, "finished": fin,
                                        "len": len(c.token_ids), "correct": score(it, c.text, fin),
                                        "text": c.text}) + "\n")
        print(f"chunk {s // CH}: {len(chunk)} prompts in {time.time() - t:.0f}s", flush=True)
    print("WORKER DONE", flush=True)

if __name__ == "__main__":
    main()
'''
(WORK / "gen_worker.py").write_text(WORKER)

# ---------- run both GPUs in parallel ----------
procs = {}
for gpu, (vid, which) in JOBS.items():
    mdir = pull(vid)
    out = WORK / f"{vid[:8]}.jsonl.gz"
    log = open(WORK / f"{vid[:8]}.log", "a")
    procs[gpu] = (vid, subprocess.Popen(
        [sys.executable, str(WORK / "gen_worker.py"), str(mdir), str(ITEMS), str(out), which, str(DEADLINE)],
        stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
    print(f"started {vid[:8]} ({which}) on GPU{gpu}", flush=True)

t0 = time.time()
while procs:
    time.sleep(60)
    for gpu, (vid, p) in list(procs.items()):
        if p.poll() is not None:
            print(f"[{time.time() - t0:5.0f}s] {vid[:8]} exit {p.returncode}", flush=True)
            if p.returncode:
                print(open(WORK / f"{vid[:8]}.log").read()[-3000:])
            del procs[gpu]

# ---------- summary + pack ----------
import gzip
from collections import defaultdict

for vid, _ in JOBS.values():
    f = WORK / f"{vid[:8]}.jsonl.gz"
    if not f.exists():
        continue
    by = defaultdict(lambda: [0, 0, 0, 0])  # n, correct, finished, tokens
    for l in gzip.open(f, "rt"):
        r = json.loads(l); b = by[r["src"]]
        b[0] += 1; b[1] += r["correct"]; b[2] += r["finished"]; b[3] += r["len"]
    print(f"\n{vid[:8]}")
    for s, (n, c, fin, tk) in by.items():
        print(f"  {s:15} n={n:5} acc={100 * c / n:5.1f} finished={100 * fin / n:5.1f} mean_len={tk / n:6.0f}")
sh("cd /kaggle/working && tar czf gen.tgz gen/*.jsonl.gz gen/train_items.json gen/*.log && ls -lh gen.tgz")
