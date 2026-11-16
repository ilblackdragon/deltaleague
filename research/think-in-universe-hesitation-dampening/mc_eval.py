#!/usr/bin/env python3
"""Official-protocol public-validation likelihood check (text MC tasks only).

PUBLIC_EVALUATION.md: public text benchmarks use native lm-eval 0.4.13 tasks,
native few-shot prompts, and lm-eval metrics; likelihood (multiple-choice) tasks
are scored from completion log-probabilities (acc / acc_norm), not by generating
letters. This script runs that protocol locally with transformers instead of a
vLLM endpoint, on exactly the public documents pinned in validation.json
(doc indices + dataset revisions).

Not included: gsm8k/math500/bbh/ifeval (generative, thinking disabled), code,
BFCL. This covers 10 of 18 public benchmarks (MLU subtasks optional).

Usage:
  python mc_eval.py --output local/mc-baseline.json
  python mc_eval.py --weights edited.safetensors --output local/mc-edit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from delta_cli import local  # noqa: E402

TASKS = ["arc_easy", "arc_challenge", "boolq", "commonsense_qa",
         "hellaswag", "openbookqa", "piqa", "winogrande"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(ROOT))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--per-task", type=int, default=None,
                    help="use the first N official doc indices per task (advisory subset)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from lm_eval import evaluator

    workspace, identity, config, metadata = local.read_workspace(Path(args.workspace))
    validation = json.loads((workspace / "validation.json").read_text())
    samples = {}
    tasks = list(args.tasks)
    for task in tasks:
        if task == "mmlu":
            for name, entry in validation["text"]["mmlu"]["selection"].items():
                samples[name] = entry["indices"]
            continue
        selection = validation["text"][task]["selection"][task]
        samples[task] = selection["indices"]
    if args.per_task:
        samples = {name: indices[:args.per_task] for name, indices in samples.items()}
    pins = validation["dataset_revisions"]

    # Freeze dataset revisions exactly as the official suite does.
    import datasets
    from unittest.mock import patch
    original = datasets.load_dataset

    def pinned(path, *a, **kw):
        if path in pins:
            kw["revision"] = pins[path]
        return original(path, *a, **kw)

    with patch.object(datasets, "load_dataset", pinned):
        with local.model_directory(workspace, metadata, weights=args.weights, patch=args.patch) as (directory, artifact):
            print(f"model directory: {directory} ({artifact['kind']})", file=sys.stderr)
            result = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={directory},dtype=float32,trust_remote_code=False",
                device=args.device,
                tasks=list(samples),
                samples=samples,
                batch_size=1,
                apply_chat_template=False,
                log_samples=True,
            )
    if result is None:
        raise SystemExit("lm_eval returned no result")

    components = {}
    for task, metrics in result["results"].items():
        value = metrics.get("acc_norm,none", metrics.get("acc,none"))
        components[task] = None if value is None else value * 100
    report = {
        "protocol": "lm-eval-0.4.13-likelihood (official public text protocol, local transformers/MPS)",
        "official_harness": False,
        "note": "Same public docs, prompts, few-shot and metrics as the pinned public suite; "
                "backend is transformers float32 on MPS instead of vLLM BF16.",
        "model_id": identity["model_id"], "source_version_id": metadata["id"],
        "config_id": config["id"], "validation_sha256": config["validation_sha256"],
        "artifact": artifact, "tasks": sorted(samples),
        "components": components,
        "mean": sum(v for v in components.values() if v is not None) / max(1, len(components)),
        "raw_results": result["results"],
        "runtime": {"device": args.device, "lm_eval": __import__("lm_eval").__version__},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, default=str, indent=2) + "\n")
    print(json.dumps({"mean": round(report["mean"], 3), "components": components}, indent=2))


if __name__ == "__main__":
    main()
