"""Decide whether a candidate passes the submission gate, and submit it if it does.

usage: gate_submit.py NAME MODEL_DIR --seeds 1000,2000 --side side_NAME --pr URL --title "..." [--dry-run]

Gate (all must hold):
  1. strict gain vs the leader's pooled samples >= MIN_GAIN with z >= MIN_Z, paired by prompt;
  2. no benchmark family loses more than its standard error;
  3. side check (code + IFEval, thinking on): truncation at most +3 points vs the leader and
     stray second </think> at most +1 point.
Writes a JSON verdict to gate/<NAME>.json. On pass and not --dry-run: delta patch build,
validate, submit, and append the response to gate/submissions.log.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

LEADER = "cd0b95bb"
LEADER_DIR = "versions/cd0b95bb-c5c0-4656-b9be-ad6482f04075"
LEADER_SEEDS = (1000, 2000, 3000)
MIN_GAIN, MIN_Z = 2.5, 2.5
DELTA = os.environ.get("DELTA_BIN", "venv/bin/delta")


def load(name, seeds):
    rows = {}
    for seed in seeds:
        path = Path(f"results/{name}_seed{seed}/rows.jsonl")
        if not path.exists():
            return None
        for line in path.open():
            row = json.loads(line)
            rows.setdefault(row["id"], []).append(row)
    return rows


def paired(cand, base, key, ids):
    diffs = [100 * (sum(r[key] for r in cand[i]) / len(cand[i]) - sum(r[key] for r in base[i]) / len(base[i])) for i in ids]
    mean = sum(diffs) / len(diffs)
    se = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs)) ** 0.5
    return mean, se


def side_stats(name):
    rows = load(name, (1000,))
    if rows is None:
        return None
    flat = [r for v in rows.values() for r in v]
    return {
        "trunc": 100 * sum(r["truncated"] for r in flat) / len(flat),
        "extra_close": 100 * sum(r["text"].count("</think>") > 1 for r in flat) / len(flat),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model")
    parser.add_argument("--seeds", default="1000,2000")
    parser.add_argument("--side", required=True)
    parser.add_argument("--pr", default="")
    parser.add_argument("--title", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    verdict = {"name": args.name, "seeds": seeds, "pass": False, "reasons": []}
    cand, base = load(args.name, seeds), load(LEADER, LEADER_SEEDS)
    if cand is None or base is None:
        verdict["reasons"].append("missing results")
    else:
        ids = sorted(cand)
        gain, se = paired(cand, base, "strict", ids)
        verdict.update({"d_strict": round(gain, 2), "se": round(se, 2), "z": round(gain / se, 2) if se else None})
        if gain < MIN_GAIN or se == 0 or gain / se < MIN_Z:
            verdict["reasons"].append(f"gain {gain:.2f} z {gain / se if se else 0:.2f} below gate")
        verdict["benchmarks"] = {}
        for bench in sorted({cand[i][0]["bench"] for i in ids}):
            sub = [i for i in ids if cand[i][0]["bench"] == bench]
            b_gain, b_se = paired(cand, base, "strict", sub)
            verdict["benchmarks"][bench] = [round(b_gain, 2), round(b_se, 2)]
            if b_gain < -b_se:
                verdict["reasons"].append(f"{bench} loses {b_gain:.2f} (se {b_se:.2f})")
        side_c, side_l = side_stats(args.side), side_stats("side_cd0b")
        verdict["side"] = {"candidate": side_c, "leader": side_l}
        if side_c is None or side_l is None:
            verdict["reasons"].append("missing side check")
        else:
            if side_c["trunc"] > side_l["trunc"] + 3:
                verdict["reasons"].append("side truncation up")
            if side_c["extra_close"] > side_l["extra_close"] + 1:
                verdict["reasons"].append("stray </think> in side check")
        verdict["pass"] = not verdict["reasons"]

    Path("gate").mkdir(exist_ok=True)
    Path(f"gate/{args.name}.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict))
    if not verdict["pass"] or args.dry_run:
        return

    quota = json.loads(subprocess.run([DELTA, "quota"], capture_output=True, text=True).stdout)
    if quota["used"] >= quota["daily_limit"]:
        print("quota exhausted")
        return
    sub = Path(f"subs/{args.name}")
    if sub.exists():
        subprocess.run(["rm", "-rf", str(sub)])
    build = subprocess.run([f"../../{DELTA}", "patch", "build", "--parent", "model", "--modified", f"../../{args.model}/model.safetensors", "--output", f"../../{sub}"],
                           cwd=LEADER_DIR, capture_output=True, text=True)
    check = subprocess.run([DELTA, "validate", str(sub)], capture_output=True, text=True)
    if build.returncode or check.returncode:
        print("build or validate failed", build.stderr[-500:], check.stderr[-500:])
        return
    command = [DELTA, "submit", str(sub), "--title", args.title] + (["--pr", args.pr] if args.pr.strip() else [])
    result = subprocess.run(command, capture_output=True, text=True)
    with open("gate/submissions.log", "a") as log:
        log.write(json.dumps({"name": args.name, "stdout": result.stdout[-3000:], "stderr": result.stderr[-1000:]}) + "\n")
    print(result.stdout[-1500:])


if __name__ == "__main__":
    main()
