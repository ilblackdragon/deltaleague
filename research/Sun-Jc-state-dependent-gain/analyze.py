"""Score a run with the platform's private-task scorer and classify how each completion ended.

Outcome classes:
  ok        finished with EOS after closing </think>
  noclose   finished with EOS without ever closing </think>
  runaway   hit the token cap while still thinking (never closed)
  overflow  closed </think>, then hit the token cap in the answer phase

`score` is delta_cli.scoring.score_answer on the raw text, which is what the platform
applies to private tasks: the answer is the text after the LAST </think>, or the whole
text when thinking never closed.

Gold answers of `math` items are wrapped as $...$ before scoring (scorelib.local_job).
`--bare-gold` scores against the pool's golds exactly as stored.

usage: python analyze.py JOBS RUN_DIR [--dump scored.jsonl] [--bare-gold]
"""
import json
import statistics
import sys

from scorelib import THINK_CLOSE, load_jobs, load_rows, outcome, score, tokenizer

jobs_path, run_dir = sys.argv[1], sys.argv[2]
dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
tok = tokenizer()
jobs = load_jobs(jobs_path, bare_gold="--bare-gold" in sys.argv)
rows = load_rows(run_dir)

scored = []
for r in rows:
    job = jobs[r["id"]]
    text = tok.decode(r["tokens"], skip_special_tokens=True)
    closes = [i for i, t in enumerate(r["tokens"]) if t == THINK_CLOSE]
    value, error = score(job, text)
    scored.append({"id": r["id"], "sample": r["sample"], "family": job["family"], "outcome": outcome(r), "score": value, "error": error,
                   "n": r["n"], "think": closes[0] if closes else r["n"], "closes": len(closes),
                   "answer_len": r["n"] - closes[-1] - 1 if closes else 0})

if dump:
    with open(dump, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")


def line(name, items):
    n = len(items)
    if not n:
        return
    c = lambda o: sum(1 for s in items if s["outcome"] == o)
    acc = sum(s["score"] for s in items) / n
    ok = [s for s in items if s["outcome"] == "ok"]
    acc_ok = sum(s["score"] for s in ok) / len(ok) if ok else float("nan")
    lost = sum(s["score"] for s in items if s["outcome"] in ("runaway", "overflow"))
    print(f"{name:9s} n={n:4d}  score {100 * acc:5.1f}  | ok {100 * c('ok') / n:5.1f}%  runaway {100 * c('runaway') / n:5.1f}%  "
          f"overflow {100 * c('overflow') / n:5.1f}%  noclose {100 * c('noclose') / n:4.1f}%  | acc(ok) {100 * acc_ok:5.1f}  "
          f"capped-but-scored {lost:.0f}  | median think {statistics.median(s['think'] for s in items):6.0f}  mean tokens {statistics.mean(s['n'] for s in items):6.0f}")


print(f"{len(scored)} completions from {run_dir}")
errors = [s for s in scored if s["error"]]
if errors:
    print(f"SCORER ERRORS on {len(errors)} completions (scored 0): {sorted({s['error'] for s in errors})}")
line("ALL", scored)
for fam in sorted({s["family"] for s in scored}):
    line(fam, [s for s in scored if s["family"] == fam])
multi = [s for s in scored if s["closes"] > 1]
print(f"completions with more than one </think>: {len(multi)} ({100 * len(multi) / max(len(scored), 1):.1f}%), "
      f"score {100 * sum(s['score'] for s in multi) / max(len(multi), 1):.1f}")
total_tokens = sum(s["n"] for s in scored)
capped_tokens = sum(s["n"] for s in scored if s["outcome"] in ("runaway", "overflow"))
print(f"tokens: {total_tokens:,} total, {100 * capped_tokens / max(total_tokens, 1):.0f}% spent in capped completions")
