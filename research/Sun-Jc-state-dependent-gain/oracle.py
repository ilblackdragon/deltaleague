"""Decode-time oracle interventions on completions that hit the token cap.

For each selected baseline completion, replay its own prefix (same prompt, same tokens,
same per-sequence random stream) and force tokens at fixed steps. Nothing is changed in
the weights: this measures the best case for a weight edit that would act at that point.

Plan syntax, comma separated, applied in order of step:
  close@N     force </think> at step N if thinking is still open
  reclose@N   force </think> at step N regardless of phase
  box@N       force the tokens of "\\n\\n\\boxed{" starting at step N
  tok@N:ID    force token ID at step N
  boxafter@N  from step N on, steer every </think> into an immediate boxed answer: force "\\n\\n"
              right after a </think>, then a backslash right after that (the model is left to
              write "boxed{...}" itself). Emulates neurons gated on "late" and on those two tokens.
  eos@N       stop at step N
  push@N:ID:D add D logits to token ID at every step >= N, except in the 8 steps after a </think>
              (a soft deadline: the model exits at its own next plausible stopping point)

usage: python oracle.py --model FP32_DIR --jobs JOBS --base-run RUN --out OUT --plan PLAN
                        [--select runaway,overflow | --late-close T] [--shard K/P] [--seed 1000]
"""
import argparse
import glob
import json
import os
import sys
import time

from engine import EOS_IDS, THINK_CLOSE, Model, Sampler, Seq, batch_limit, run_batch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--base-run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--select", default="runaway,overflow")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--kv-gb", type=float, default=1.0)
ap.add_argument("--late-close", type=int, default=-1,
                help="instead of --select: take every completion with a </think> at step >= T and replay it up to that </think>")
args = ap.parse_args()

k, p = (int(x) for x in args.shard.split("/"))
os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"shard-{k}.jsonl")
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        try:
            r = json.loads(line)
            done.add((r["id"], r["sample"]))
        except (json.JSONDecodeError, KeyError):
            pass

model = Model(args.model).install_fast_paths()
sampler = Sampler(max_tokens=args.max_tokens)
kv_budget = int(args.kv_gb * 2**30)
jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
box_tokens = model.tok("\n\n\\boxed{", add_special_tokens=False).input_ids
NEWLINES, BACKSLASH = box_tokens[0], box_tokens[1]
plan, pushes = [], []
for item in args.plan.split(","):
    kind, where = item.split("@")
    at, _, token = where.partition(":")
    if kind == "push":
        token_id, delta = token.split(":")
        pushes.append((int(at), int(token_id), float(delta)))
        plan.append((int(at), kind, None))
    else:
        plan.append((int(at), kind, int(token) if token else None))
plan.sort(key=lambda x: x[0])
first_step = plan[0][0]


def bias(s, step):
    """Soft pushes, suspended for the 8 steps after a </think> so the boxed answer can be written."""
    if THINK_CLOSE in s.out[-8:]:
        return None
    return [(token_id, delta) for at, token_id, delta in pushes if step >= at]
select = set(args.select.split(","))

rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(args.base_run, "shard-*.jsonl"))) for l in open(f) if l.strip()]
pending = []
index = 0
for r in rows:
    closes = THINK_CLOSE in r["tokens"]
    outcome = ("ok" if closes else "noclose") if r["finish"] == "stop" else ("overflow" if closes else "runaway")
    if args.late_close >= 0:
        cut = next((i + 1 for i, t in enumerate(r["tokens"]) if t == THINK_CLOSE and i >= args.late_close), None)
        if cut is None:
            continue
    elif outcome not in select or r["n"] <= first_step:
        continue
    else:
        cut = first_step
    mine = index % p == k
    index += 1
    if not mine or (r["id"], r["sample"]) in done:
        continue
    prompt = model.encode_prompt(jobs[r["id"]]["prompt"])
    s = Seq(f"{r['id']}#{r['sample']}", prompt, args.seed, args.max_tokens,
            meta={"id": r["id"], "sample": r["sample"], "base_outcome": outcome, "applied": []}, out=r["tokens"][:cut])
    pending.append(s)


def hook(s, step, token):
    """Return a replacement token for this step, or None to keep the sampled one."""
    for at, kind, token_id in plan:
        if kind == "boxafter" and step >= at and s.out:
            if s.out[-1] == THINK_CLOSE:
                s.meta["applied"].append(f"nl@{step}")
                return NEWLINES
            if len(s.out) >= 2 and s.out[-2] == THINK_CLOSE and s.out[-1] == NEWLINES:
                s.meta["applied"].append(f"backslash@{step}")
                return BACKSLASH
        if kind == "tok" and step == at:
            s.meta["applied"].append(f"tok@{at}:{token_id}")
            return token_id
        if kind == "close" and step == at and THINK_CLOSE not in s.out:
            s.meta["applied"].append(f"close@{at}")
            return THINK_CLOSE
        if kind == "reclose" and step == at:
            s.meta["applied"].append(f"reclose@{at}")
            return THINK_CLOSE
        if kind == "box" and at <= step < at + len(box_tokens):
            if step == at:
                s.meta["applied"].append(f"box@{at}")
            return box_tokens[step - at]
        if kind == "eos" and step == at:
            s.meta["applied"].append(f"eos@{at}")
            return EOS_IDS[0]
    return None


t0 = time.time()
sink = open(out_path, "a")
print(f"[oracle {k}/{p}] plan {args.plan}: {len(pending)} completions", file=sys.stderr, flush=True)


def on_finish(s):
    rec = {"id": s.meta["id"], "sample": s.meta["sample"], "tokens": s.out, "finish": s.finish, "n": len(s.out),
           "closed": THINK_CLOSE in s.out, "base_outcome": s.meta["base_outcome"], "applied": s.meta["applied"], "plan": args.plan}
    sink.write(json.dumps(rec) + "\n")
    sink.flush()


while pending:
    pending.sort(key=lambda s: s.total())
    n = min(args.batch, len(pending))
    while n > 2 and n > batch_limit(pending[n - 1].total(), args.batch, kv_budget):
        n -= 1
    batch, pending = pending[:n], pending[n:]
    size = len(batch)
    pending.extend(run_batch(model, batch, sampler, lambda live: len(pending) >= size - live, on_finish, kv_budget, hook=hook, bias=bias if pushes else None))
print(f"[oracle {k}/{p}] finished in {(time.time() - t0) / 60:.1f} min", file=sys.stderr, flush=True)
sink.close()
