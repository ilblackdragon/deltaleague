"""Build a ~510-item local proxy of the private test (GSM8K / MATH-500 / ARC-Challenge)."""
import json, random
from datasets import load_dataset
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
rng = random.Random(1108)
items = []
g = list(load_dataset("openai/gsm8k", "main", split="test"))
for i, r in enumerate(rng.sample(g, 170)):
    items.append({"id": f"gsm-{i}", "family": "gsm", "scorer": "math",
                  "prompt": r["question"] + SUF, "answer": r["answer"].split("####")[-1].strip().replace(",", "")})
m = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
for i, r in enumerate(rng.sample(m, 170)):
    items.append({"id": f"math-{i}", "family": "math", "scorer": "math", "prompt": r["problem"] + SUF, "answer": r["answer"]})
a = [r for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if len(r["choices"]["label"]) == 4]
for i, r in enumerate(rng.sample(a, 170)):
    labels = r["choices"]["label"]; texts = r["choices"]["text"]
    ans = "ABCD"[labels.index(r["answerKey"])]
    body = r["question"] + "\n" + "\n".join(f"{L}. {t}" for L, t in zip("ABCD", texts))
    items.append({"id": f"arc-{i}", "family": "arc", "scorer": "choice", "prompt": body + SUF, "answer": ans})
json.dump(items, open("runs/proxy.json", "w"), indent=1)
print(len(items))
