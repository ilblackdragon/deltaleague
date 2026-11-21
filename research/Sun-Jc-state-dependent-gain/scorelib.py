"""Shared helpers for scoring local runs with the platform's private-task scorer.

`score_answer` (delta_cli.scoring) is used unchanged. The only local adaptation is the
gold format of `math` items: the scorer parses the gold with math-verify exactly like a
model answer, and a bare LaTeX gold such as `8x^9`, `\\tan x` or `(6,7)` has no `$...$` or
`\\boxed{}` anchor, so it parses to nothing or to a stray number and a correct model
answer scores 0. `local_job` wraps such golds as `$...$`.
"""
import glob
import json
import os

from delta_cli.scoring import score_answer

THINK_CLOSE = 248069


def tokenizer(path=None):
    """Tokenizer for decoding stored token ids: any checkpoint directory of the competition model."""
    from transformers import AutoTokenizer
    path = path or os.environ.get("DELTA_LAB_MODEL", "model")  # a `delta competition init` workspace has model/
    return AutoTokenizer.from_pretrained(os.path.expanduser(path), local_files_only=True)


def local_job(job, bare_gold=False):
    """The job as scored locally: math golds get a $...$ anchor unless they already have one."""
    gold = str(job.get("answer", ""))
    if bare_gold or job.get("scorer") != "math" or "$" in gold or "\\boxed" in gold:
        return job
    return dict(job, answer=f"${gold}$")


def load_jobs(path, bare_gold=False):
    return {j["id"]: local_job(j, bare_gold) for j in map(json.loads, open(os.path.expanduser(path)))}


def load_rows(run_dir):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(run_dir), "shard-*.jsonl")))
    return [json.loads(line) for f in files for line in open(f) if line.strip()]


def outcome(row):
    """ok / noclose (stopped without closing) / runaway (cap while thinking) / overflow (cap after closing)."""
    closed = THINK_CLOSE in row["tokens"]
    if row["finish"] == "stop":
        return "ok" if closed else "noclose"
    return "overflow" if closed else "runaway"


def score(job, text):
    """(score, error name or None). Scorer exceptions are reported, never silently scored."""
    if job.get("scorer") == "ifeval":
        # IFEval's language check uses langdetect, which is randomised unless seeded; seed it so
        # that re-scoring the same text always gives the same local result.
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
    try:
        return float(score_answer(job, text)), None
    except Exception as exc:
        return 0.0, type(exc).__name__
