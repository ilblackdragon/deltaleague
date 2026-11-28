"""Option B: build preference pairs (chosen = finished correct, rejected = hit the 8192 cap).

Rejected traces are cut so a pair fits T4 memory and still contains where it went wrong:
after-</think> truncations keep think + REJ_AFTER answer tokens; in-think ones keep REJ_MAX tokens.
Usage: python make_pairs.py GEN_DIR OUT.jsonl
"""
import collections, glob, gzip, json, random, sys
from pathlib import Path
from transformers import AutoTokenizer

gen, out = Path(sys.argv[1]), sys.argv[2]
REJ_AFTER, REJ_CAP, REJ_MAX, MAX_CHOSEN, MAX_ANS, PER_ITEM = 1024, 6144, 4096, 6000, 2000, 3
tok = AutoTokenizer.from_pretrained(str(Path(__file__).parent / "model-hf"))
items = {i["id"]: i for i in json.load(open(gen / "train_items.json"))}
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
by = collections.defaultdict(list)
for f in glob.glob(str(gen / "shard*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        r = json.loads(l); by[r["id"]].append(r)

rng = random.Random(0)
pairs, st = [], collections.Counter()
for iid, rs in by.items():
    good = []
    for r in rs:
        if not (r["correct"] and r["finished"] and "</think>" in r["text"]): continue
        ans = r["text"].split("</think>")[-1]
        if "\\boxed{" in ans and r["len"] <= MAX_CHOSEN and len(tok.encode(ans)) <= MAX_ANS:
            good.append(r)
    bad = [r for r in rs if not r["finished"]]
    if not good or not bad: continue
    good.sort(key=lambda r: r["len"])
    bad.sort(key=lambda r: "</think>" not in r["text"])  # answer-phase failures first
    prompt = tok.apply_chat_template([{"role": "user", "content": items[iid]["q"] + SUFFIX}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
    for k, b in enumerate(bad[:PER_ITEM]):
        c = good[(len(good) // 2 + k) % len(good)]
        ids = tok.encode(b["text"])
        if "</think>" in b["text"]:
            n = len(tok.encode(b["text"].split("</think>")[0] + "</think>")) + REJ_AFTER
            n = min(n, REJ_CAP); st["after_think"] += 1
        else:
            n = REJ_MAX; st["in_think"] += 1
        pairs.append({"id": iid, "src": items[iid]["src"], "level": items[iid].get("level"), "prompt": prompt,
                      "chosen": c["text"] + "<|im_end|>", "rejected": tok.decode(ids[:n]),
                      "chosen_len": c["len"], "rejected_len": min(n, len(ids))})
rng.shuffle(pairs)
with open(out, "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
print(len(pairs), "pairs from", len({p["id"] for p in pairs}), "items", dict(st))
print("levels", sorted(collections.Counter(str(p["level"]) for p in pairs).items()))
m = lambda k: sorted(p[k] for p in pairs)[len(pairs) // 2]
print("median chosen", m("chosen_len"), "rejected", m("rejected_len"), "total tokens", sum(p["chosen_len"] + p["rejected_len"] for p in pairs))
