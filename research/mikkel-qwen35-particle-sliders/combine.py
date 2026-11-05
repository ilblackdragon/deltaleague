"""Combine disjoint likelihood reports without rerunning model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(inputs, output):
    reports = [json.loads(path.read_text()) for path in inputs]
    if output.exists() or output.with_suffix(".generations.jsonl").exists():
        raise FileExistsError(output)
    result = dict(reports[0])
    scores, selected, metrics, names, rows = {}, {}, {}, [], []
    for path, report in zip(inputs, reports):
        for key in ("model_id", "config_id", "source_version_id", "generation", "artifact", "runtime", "official_manifest_sha256"):
            if report[key] != result[key]:
                raise ValueError("Reports do not describe the same candidate/protocol: " + key)
        if report["schema"] != "qwen35-local-likelihood-advisory-v1":
            raise ValueError("Only disjoint native likelihood reports can be combined")
        if scores.keys() & report["validation_benchmarks"].keys():
            raise ValueError("Benchmark sets overlap")
        scores.update(report["validation_benchmarks"])
        selected.update(report["protocol"]["selected"])
        metrics.update(report["protocol"]["task_metrics"])
        names.extend(report["coverage"]["benchmarks"])
        rows.extend(path.with_suffix(".generations.jsonl").read_text().splitlines())
    result["protocol"] = dict(selected=selected,chat_template=False,max_length=2048,task_metrics=metrics)
    result["validation_sha256"] = hashlib.sha256(json.dumps(result["protocol"],sort_keys=True).encode()).hexdigest()
    result["validation_benchmarks"] = scores
    result["validation_score"] = statistics.mean(scores.values())
    result["coverage"] = dict(benchmarks=names,examples=len(rows),full_official_suite=False)
    result["derived_from"] = [{"report_sha256":file_hash(path),"rows_sha256":file_hash(path.with_suffix(".generations.jsonl"))} for path in inputs]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    output.with_suffix(".generations.jsonl").write_text("\n".join(rows)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs",nargs="+",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(combine(args.inputs,args.output)["validation_benchmarks"],indent=2))
