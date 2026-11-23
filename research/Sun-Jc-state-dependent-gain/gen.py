"""Sample completions for a job file with the evaluation sampler. Shardable and resumable.

jobs.jsonl: one object per line with at least {"id", "prompt"} ("prompt" is the user message).
Output: OUT/shard-K.jsonl, one line per finished completion:
  {"id", "sample", "tokens", "finish" ("stop" | "length"), "n", "closed", "secs"}

usage: python gen.py --model FP32_DIR --jobs jobs.jsonl --out OUT --shard K/P [--samples N]
                     [--seed S] [--batch 16] [--max-tokens 8192] [--kv-gb 1.2] [--no-thinking]
                     [--gain G] [--edit EDITS.pt]
"""
import argparse
import json
import os
import sys
import time

from engine import THINK_CLOSE, Model, Sampler, Seq, batch_limit, run_batch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--kv-gb", type=float, default=1.2)
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--gain", type=float, default=1.0,
                help="emulate scaling the final-norm gain by this factor: logits g*z under the fixed evaluation "
                     "sampler equal temperature 1/g with presence penalty 1.5/g on the unscaled logits")
ap.add_argument("--edit", default="", help="edit records (edits.py) applied to the model in memory before sampling")
args = ap.parse_args()

k, p = (int(x) for x in args.shard.split("/"))
os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"shard-{k}.jsonl")
done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["id"], r["sample"]))
            except (json.JSONDecodeError, KeyError):
                pass  # a partial last line from an interrupted run

jobs = [json.loads(line) for line in open(args.jobs) if line.strip()]
model = Model(args.model).install_fast_paths()
if args.edit:
    import edits
    edits.apply_to_model(model, edits.load(args.edit))
sampler = Sampler(temperature=1.0 / args.gain, presence_penalty=1.5 / args.gain, max_tokens=args.max_tokens)
kv_budget = int(args.kv_gb * 2**30)

pending = []
index = 0
for job in jobs:
    for s in range(args.samples):
        mine = index % p == k
        index += 1
        if mine and (job["id"], s) not in done:
            prompt = model.encode_prompt(job["prompt"], thinking=not args.no_thinking)
            pending.append(Seq(f"{job['id']}#{s}", prompt, args.seed, args.max_tokens, meta={"id": job["id"], "sample": s}))

total, t0, tokens_done = len(pending), time.time(), 0
started = {}
sink = open(out_path, "a")
print(f"[shard {k}/{p}] {total} completions to generate ({len(done)} already done)", file=sys.stderr, flush=True)


def on_finish(s):
    global tokens_done
    tokens_done += len(s.out)
    rec = {"id": s.meta["id"], "sample": s.meta["sample"], "tokens": s.out, "finish": s.finish,
           "n": len(s.out), "closed": THINK_CLOSE in s.out, "secs": round(time.time() - started.get(s.key, t0), 1)}
    sink.write(json.dumps(rec) + "\n")
    sink.flush()


finished = 0
last_report = t0
gen_tokens_at_start = sum(len(s.out) for s in pending)
while pending:
    pending.sort(key=lambda s: s.total())
    n = min(args.batch, len(pending))
    while n > 2 and n > batch_limit(pending[n - 1].total(), args.batch, kv_budget):
        n -= 1
    batch, pending = pending[:n], pending[n:]
    for s in batch:
        started.setdefault(s.key, time.time())
    size = len(batch)
    suspended = run_batch(model, batch, sampler, lambda live: len(pending) >= size - live, on_finish, kv_budget)
    finished += size - len(suspended)
    pending.extend(suspended)
    now = time.time()
    if now - last_report > 60 or not pending:
        made = tokens_done + sum(len(s.out) for s in pending) - gen_tokens_at_start
        print(f"[shard {k}/{p}] {finished}/{total} done, {len(pending)} pending, "
              f"{made / max(now - t0, 1e-9):.1f} tok/s, {(now - t0) / 60:.1f} min", file=sys.stderr, flush=True)
        last_report = now
sink.close()
