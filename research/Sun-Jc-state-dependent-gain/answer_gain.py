"""Ceiling experiment for a logit gain confined to the answer phase.

For each stored completion that closed its thinking, the prompt and the stored tokens up to
and including the first </think> are replayed once (teacher forced). The rest is then sampled
several times in one batch: every row applies the evaluation sampler to the logits multiplied
by its own gain, which is exactly what a final-norm gain that is active only after </think>
would produce. The thinking is the stored one for every row, so differences between gains
come from the answer phase alone.

Row (gain 1, draw 0) uses the stored random stream and must reproduce the stored completion;
the share of exact reproductions is reported as a check of the replay. Draw j of every gain
shares one random stream, so gains are compared on common random numbers.

usage: python answer_gain.py --model FP32_DIR --jobs JOBS --base-run RUN --out OUT
           [--gains 1,1.25,1.5,2,3] [--draws 4] [--families ifeval,math] [--outcomes ok]
           [--shard K/P] [--seed 1000] [--kv-gb 1.0] [--max-answer 1000] [--stored-answer-max N] [--edit EDITS.pt]

With --edit the continuations are written by the edited model (use --gains 1): the same paired
design then measures the answer-phase effect of an actual weight edit.
Output: OUT/shard-K.jsonl, one line per (completion, gain, draw):
  {"id", "sample", "gain", "draw", "close", "answer" (token ids after the first </think>), "finish", "n"}
"""
import argparse
import json
import os
import sys
import time

import torch

from engine import EOS_IDS, KV_BYTES_PER_TOKEN, THINK_CLOSE, Model, Sampler, uniform_stream
from scorelib import load_rows, outcome

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--base-run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gains", default="1,1.25,1.5,2,3")
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--families", default="")
ap.add_argument("--outcomes", default="ok")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--kv-gb", type=float, default=1.0)
ap.add_argument("--edit", default="", help="edit records applied to the model that writes the continuations (the stored thinking is still the parent's)")
ap.add_argument("--stored-answer-max", type=int, default=0, help="only completions whose stored answer has at most this many tokens (0: all)")
ap.add_argument("--max-answer", type=int, default=1000,
                help="stop a continuation after this many answer tokens (finish 'length'): a rambling answer would "
                     "otherwise run to the 8,192 cap and dominate the run time; it fails either way")
args = ap.parse_args()

k, p = (int(x) for x in args.shard.split("/"))
os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"shard-{k}.jsonl")
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        try:
            r = json.loads(line)
            done.add((r["id"], r["sample"], r["gain"], r["draw"]))
        except (json.JSONDecodeError, KeyError):
            pass  # a partial last line from an interrupted run

gains = [float(g) for g in args.gains.split(",")]
arms = [(g, j) for j in range(args.draws) for g in gains]
families = set(filter(None, args.families.split(",")))
outcomes = set(args.outcomes.split(","))
jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
rows = sorted(load_rows(args.base_run), key=lambda r: (r["n"], r["id"], r["sample"]))
rows = [r for r in rows if THINK_CLOSE in r["tokens"] and outcome(r) in outcomes
        and (not families or jobs[r["id"]].get("family", jobs[r["id"]].get("scorer")) in families)]
if args.stored_answer_max:
    rows = [r for r in rows if r["n"] - r["tokens"].index(THINK_CLOSE) - 1 <= args.stored_answer_max]
rows = [r for i, r in enumerate(rows) if i % p == k]

model = Model(args.model).install_fast_paths()
if args.edit:
    import edits
    edits.apply_to_model(model, edits.load(args.edit))
sampler = Sampler(max_tokens=args.max_tokens)
kv_budget = int(args.kv_gb * 2**30)
sink = open(out_path, "a")
t0, reproduced, checked = time.time(), 0, 0
print(f"[answer_gain {k}/{p}] {len(rows)} completions x {len(arms)} rows (gains {gains}, {args.draws} draws)", file=sys.stderr, flush=True)


@torch.inference_mode()
def continue_rows(prefix_ids, generated_prefix, key, todo):
    """Sample one continuation per (gain, draw) in `todo` from a shared prefix. Returns {(gain, draw): (tokens, finish)}."""
    past, mask, h = model.prefill([prefix_ids])
    b = len(todo)
    index = torch.zeros(b, dtype=torch.long)
    past.reorder_cache(index)
    mask, h = mask[index], h[index]
    seen = torch.zeros((b, model.vocab), dtype=torch.bool)
    seen[:, torch.tensor(sorted(set(generated_prefix)), dtype=torch.long)] = True
    gain = torch.tensor([g for g, _ in todo], dtype=torch.float32)
    streams = {j: uniform_stream(args.seed if j == 0 else f"{args.seed}/draw{j}", key, args.max_tokens + 1) for _, j in todo}
    live = list(range(b))
    outs, result = [[] for _ in todo], {}
    step = len(generated_prefix)  # index of the next generated token, as in the stored completion
    while live:
        u = torch.tensor([float(streams[todo[i][1]][step]) for i in live], dtype=torch.float32)
        tokens, _ = sampler.sample(model.head(h) * gain[:, None], seen, u)
        keep = []
        for pos, i in enumerate(live):
            t = int(tokens[pos])
            finish = None
            if t in EOS_IDS:
                finish = "stop"
            else:
                outs[i].append(t)
                if step + 1 >= args.max_tokens or len(outs[i]) >= args.max_answer:
                    finish = "length"
            if finish is None:
                keep.append(pos)
            else:
                result[todo[i]] = (outs[i], finish)
        if not keep:
            break
        if len(keep) < len(live):
            sel = torch.tensor(keep, dtype=torch.long)
            past.reorder_cache(sel)
            mask, seen, tokens, gain = mask[sel], seen[sel], tokens[sel], gain[sel]
            live = [live[pos] for pos in keep]
        seen[torch.arange(len(live)), tokens] = True
        past, mask, h = model.step(tokens, past, mask)
        step += 1
    return result


for ri, r in enumerate(rows):
    close = r["tokens"].index(THINK_CLOSE)
    generated_prefix = r["tokens"][:close + 1]
    prefix_ids = model.encode_prompt(jobs[r["id"]]["prompt"]) + generated_prefix
    todo = [a for a in arms if (r["id"], r["sample"], a[0], a[1]) not in done]
    # Rows that fit the key/value budget if every continuation ran to the cap would be too few;
    # budget for the prefix plus 1,024 more tokens and split the arms into groups of that size.
    group = max(2, kv_budget // ((len(prefix_ids) + 1024) * KV_BYTES_PER_TOKEN))
    for a in range(0, len(todo), group):
        for (g, j), (tokens, finish) in continue_rows(prefix_ids, generated_prefix, f"{r['id']}#{r['sample']}", todo[a:a + group]).items():
            if g == 1.0 and j == 0 and not args.edit:
                checked += 1
                reproduced += tokens == r["tokens"][close + 1:] and finish == r["finish"]
            sink.write(json.dumps({"id": r["id"], "sample": r["sample"], "gain": g, "draw": j, "close": close,
                                   "answer": tokens, "finish": finish, "n": close + 1 + len(tokens)}) + "\n")
        sink.flush()
    if (ri + 1) % 5 == 0 or ri + 1 == len(rows):
        print(f"[answer_gain {k}/{p}] {ri + 1}/{len(rows)} completions, {(time.time() - t0) / 60:.1f} min, "
              f"stored completion reproduced {reproduced}/{checked}", file=sys.stderr, flush=True)
sink.close()
