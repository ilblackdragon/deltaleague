#!/usr/bin/env python3
"""Advisory local comparison on public validation docs (not the official harness).

The CLI's `delta evaluate` (deltaleague 0.1.1) rejects the current public
validation.json because the server now ships a selection-only format
(`text`/`code` groups) while the CLI expects benchmark -> examples. This
harness rebuilds a small, fixed subset of the *same* public docs from the
pinned dataset revisions and scores them with the CLI's own scorer
(`delta_cli.scoring.score_answer`), so baseline and candidate are compared
under identical prompts, sampler settings, seeds, and device.

Deviations from the official runner, disclosed in the report:
  * gsm8k is 0-shot here (official: 5-shot); MC tasks are 0-shot (matches).
  * plain prompts, no lm-eval few-shot rendering.
  * transformers/MPS float32 instead of vLLM/BF16.
Absolute scores are therefore not official; only paired differences matter.

Usage:
  python local_eval.py --output local/baseline.json
  python local_eval.py --weights edited.safetensors --output local/edit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_cli import local  # noqa: E402
from delta_cli.scoring import score_answer  # noqa: E402

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "210d026faf9955653af8916fad021475a3f00453"),
    "gsm8k": ("openai/gsm8k", "main", "test", "740312add88f781978c0658806c59bc2815b9866"),
}


def arc_examples(n: int, seed_index: int = 0):
    from datasets import load_dataset
    name, subset, split, revision = DATASETS["arc_easy"]
    ds = load_dataset(name, subset, split=split, revision=revision)
    validation = json.loads((ROOT / "validation.json").read_text())
    indices = validation["text"]["arc_easy"]["selection"]["arc_easy"]["indices"]
    chosen = indices[seed_index:seed_index + n]
    rows = []
    for idx in chosen:
        doc = ds[idx]
        labels = doc["choices"]["label"]
        texts = doc["choices"]["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        prompt = (f"Question: {doc['question']}\n{options}\n"
                  "Answer with the letter of the correct option.")
        rows.append({"id": f"arc_easy:{idx}", "benchmark": "arc_easy", "prompt": prompt,
                     "example": {"answer": doc["answerKey"], "scorer": "choice"}})
    return rows


def gsm8k_examples(n: int, seed_index: int = 0):
    from datasets import load_dataset
    name, subset, split, revision = DATASETS["gsm8k"]
    ds = load_dataset(name, subset, split=split, revision=revision)
    validation = json.loads((ROOT / "validation.json").read_text())
    indices = validation["text"]["gsm8k"]["selection"]["gsm8k"]["indices"]
    chosen = indices[seed_index:seed_index + n]
    rows = []
    for idx in chosen:
        doc = ds[idx]
        gold = doc["answer"].split("####")[-1].strip()
        prompt = (f"{doc['question']}\n\n"
                  "Solve the problem. Put the final answer in \\boxed{}.")
        rows.append({"id": f"gsm8k:{idx}", "benchmark": "gsm8k", "prompt": prompt,
                     "example": {"answer": gold, "scorer": "math"}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--benchmark", action="append", choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--no-thinking", action="store_true", help="disable the thinking trace for this local run")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    workspace, identity, config, metadata = local.read_workspace(Path(args.workspace))
    generation = dict(config["generation"])
    generation["max_tokens"] = args.max_new_tokens
    if args.no_thinking:
        generation["enable_thinking"] = False
    benchmarks = args.benchmark or ["arc_easy", "gsm8k"]
    tasks = []
    for name in benchmarks:
        tasks += (arc_examples if name == "arc_easy" else gsm8k_examples)(args.limit)

    with local.model_directory(workspace, metadata, weights=args.weights, patch=args.patch) as (directory, artifact):
        runner = local.Runner(directory, args.device)
        records = []
        for i, task in enumerate(tasks, 1):
            result = runner.generate(task["prompt"], generation, args.seed_base)
            score = score_answer(task["example"], result["text"])
            answer = result["text"].split("</think>")[-1]
            records.append({
                "id": task["id"], "benchmark": task["benchmark"], "example_index": i - 1,
                "answer": task["example"]["answer"], "score": score,
                "truncated": result["truncated"], "tokens": result["tokens"],
                "hesitation_tokens": len(re.findall(r"\b(?:Wait|Hmm|Actually|Alternatively|Perhaps|Maybe)\b", answer)),
                "final_answer": answer.strip()[-160:],
            })
            print(f"[{i}/{len(tasks)}] {task['id']} score={score} tokens={result['tokens']} "
                  f"truncated={result['truncated']}", file=sys.stderr)

    by_bench = {}
    for name in benchmarks:
        rows = [r for r in records if r["benchmark"] == name]
        by_bench[name] = {
            "n": len(rows),
            "score": sum(r["score"] for r in rows) / len(rows) * 100,
            "truncation_rate": sum(r["truncated"] for r in rows) / len(rows),
            "mean_tokens": sum(r["tokens"] for r in rows) / len(rows),
        }
    report = {
        "advisory": True,
        "official_harness": False,
        "note": "Rebuilt from the same public docs; gsm8k is 0-shot locally (official 5-shot). "
                "Compare paired differences, not absolute values.",
        "model_id": identity["model_id"], "source_version_id": metadata["id"],
        "config_id": config["id"], "validation_sha256": config["validation_sha256"],
        "artifact": artifact, "runtime": runner.details,
        "generation": generation, "seed_base": args.seed_base,
        "thinking": bool(generation.get("enable_thinking", True)),
        "benchmarks": by_bench,
        "overall_score": sum(v["score"] for v in by_bench.values()) / len(by_bench),
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("benchmarks", "overall_score")}, indent=2))


if __name__ == "__main__":
    main()
