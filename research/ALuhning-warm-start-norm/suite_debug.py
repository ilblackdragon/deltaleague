"""Run the pinned public suite like `delta public-eval run`, but print the full traceback on failure (for diagnosing
the local math500 AssertionError). Same arguments and checks as delta_cli/public_eval/cli.py run_command.
Usage: python suite_debug.py URL MODEL TOKENIZER CONTENT_HASH OUT MANIFEST EVAL_CONFIG"""
import json, sys, traceback
from pathlib import Path
from delta_cli.public_eval.runner import evaluate_public
from delta_cli.artifacts import file_hash
url, model, tok, h, out, manifest, cfgp = sys.argv[1:8]
cfg = json.load(open(cfgp))
assert cfg["protocol_version"] == "delta-public-v1" and file_hash(Path(manifest)) == cfg["validation_sha256"]
try:
    r = evaluate_public(endpoint=url, model=model, tokenizer=Path(tok), content_hash=h, output=Path(out),
                        manifest_path=Path(manifest), concurrency=cfg.get("generation", {}).get("max_num_seqs", 32),
                        generation=cfg["generation"]["public_validation"], config_id=cfg["id"],
                        progress=lambda d, t: print(f"Public validation: {d}/{t}", file=sys.stderr, flush=True))
    print("SUITE OK", json.dumps(r)[:300] if not isinstance(r, (str, Path)) else r)
except Exception:
    print("SUITE TRACEBACK BEGIN"); traceback.print_exc(file=sys.stdout); print("SUITE TRACEBACK END"); sys.exit(1)
