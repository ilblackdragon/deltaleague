"""Practice data from TRAINING splits only (never public validation or MATH-500 test problems).
gsm8k: openai/gsm8k main/train. MATH: EleutherAI/hendrycks_math train (levels 1-4), boxed answer extracted.
Writes data/train.jsonl and data/heldout.jsonl rows {bench, id, question, answer}. Seed 20261124.
Usage: python prep_data.py OUT_DIR"""
import json, os, random, sys
from datasets import load_dataset
out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = random.Random(20261124)
def boxed(s):
    i = s.rfind("\\boxed{");
    if i < 0: return None
    j, d = i + 7, 1
    while j < len(s) and d: d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + 7:j - 1] if d == 0 else None
rows = []
for k, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
    rows.append({"bench": "gsm8k", "id": f"gsm8k-train-{k}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")})
subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
for s in subjects:
    for k, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="train")):
        a = boxed(r["solution"])
        if a and r["level"] in ("Level 1", "Level 2", "Level 3", "Level 4"):
            rows.append({"bench": "math", "id": f"math-train-{s}-{k}", "question": r["problem"], "answer": a, "level": r["level"]})
rng.shuffle(rows)
g = [r for r in rows if r["bench"] == "gsm8k"]; m = [r for r in rows if r["bench"] == "math"]
held = g[:64] + m[:64]; train = g[64:] + m[64:]; rng.shuffle(train)
for name, data in (("heldout", held), ("train", train)):
    with open(f"{out}/{name}.jsonl", "w") as f:
        for r in data: f.write(json.dumps(r) + "\n")
print(len(train), "train;", len(held), "heldout (64 gsm8k + 64 MATH L1-4)")
