"""Public-suite groups (text + code; BFCL skipped) against a served checkpoint, resumable per group.
IFEval: the pinned doc digests do not match in this environment, so its digest check is skipped;
parent and candidate see the same docs, so the comparison stays paired.
usage: pe_groups.py ENDPOINT TOKENIZER OUTDIR VALIDATION.json EVALUATION.json"""
import json, sys
from pathlib import Path
import delta_cli.public_eval.text as text
from delta_cli.public_eval.suite import PRIMARY
endpoint, tokenizer, out, vpath, epath = sys.argv[1:6]
out = Path(out); (out / "reports").mkdir(parents=True, exist_ok=True)
suite = json.load(open(vpath)); gen = json.load(open(epath))["generation"]["public_validation"]
orig = text.verify_tasks
def lenient(tasks, selection):
    try: orig(tasks, selection)
    except ValueError as e:
        if "ifeval" not in str(e): raise
        print("WARNING:", e, "(digest check skipped)", flush=True)
text.verify_tasks = lenient
scores = json.loads((out / "scores.json").read_text()) if (out / "scores.json").exists() else {}
for group, spec in suite["text"].items():
    rep = out / "reports" / (group + ".json")
    if not rep.exists():
        text.run(suite, group, endpoint, "candidate", tokenizer, rep, concurrency=128, generation=gen)
    report = json.loads(rep.read_text()); metric, flt = PRIMARY[group]
    items = [float(s[metric]) for t in spec["selection"] for s in report["samples"][t] if s["filter"] == flt]
    scores[group] = 100 * sum(items) / len(items); (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(group, round(scores[group], 2), len(items), flush=True)
from delta_cli.public_eval.coding import run as code_run
for group in suite["code"]:
    rep = out / "reports" / (group + ".json")
    if not rep.exists():
        code_run(suite, group, endpoint, "candidate", "", rep, sandbox_backend="wasi", concurrency=128, generation=gen)
    report = json.loads(rep.read_text())
    scores[group] = 100 * sum(float(r["passed"]) for r in report["samples"]) / len(report["samples"])
    (out / "scores.json").write_text(json.dumps(scores, indent=1)); print(group, round(scores[group], 2), flush=True)
print("MEAN", round(sum(scores.values()) / len(scores), 2), "over", len(scores), "groups", flush=True)
