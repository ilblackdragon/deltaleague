"""Prepare a fixed, disjoint development set and summarize paired pilot scores."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from math import comb

from r2_data import canonical, rows, write

SEED = 20261127
MIX = Path("runs/r2-round2-v3/generation/mix.jsonl")
MIX_HASH = "8488c9c04de15e019e88fee6ef5eadc8315a109d9529373097c265b0e2521a39"


def instructions():
    topics = [
        "organizing a school seed library", "preparing an outdoor chess tournament",
        "hosting a community radio open house", "planning an accessible night-sky walk",
        "running a bicycle-light repair booth", "starting a neighborhood tool inventory",
        "setting up a weekend bird-count project", "preparing a low-waste cookbook swap",
    ]
    for index, topic in enumerate(topics):
        options = (
            ("Use exactly three lines, each starting with '- '. Include the word 'checklist' exactly once.",
             {"format": "bullets", "count": 3, "keyword": "checklist"}),
            ("Use the standalone headings 'Setup' and 'Review', each followed by one paragraph on a single line of at least eight words. Include the word 'clear' at least once.",
             {"format": "sections", "headings": ["Setup", "Review"], "keyword": "clear"}),
            ("Start with the standalone line 'Start here' and finish with the standalone line 'All set'. Write 45 to 85 words between those lines, without the word 'easy'.",
             {"format": "bounded", "start": "Start here", "end": "All set", "forbidden": "easy"}),
            ("Return only a JSON object with exactly the keys 'setup' and 'fallback'. Each value must be a string with at least eight words.",
             {"format": "json", "keys": ["setup", "fallback"]}),
        )
        for variant, (constraint, spec) in enumerate(options):
            question = f"Give actionable advice for {topic}. {constraint}"
            yield {
                "id": f"pilot-dev-instruction-{index}-{variant}", "bench": "instructions",
                "family": "instruction", "question": question, "prompt_text": question, "answer": spec,
            }


def prepare(out):
    from transformers import AutoTokenizer

    if hashlib.sha256(MIX.read_bytes()).hexdigest() != MIX_HASH:
        raise ValueError("Phase B training pool checksum differs")
    pool = rows(MIX)
    for entry in pool:
        if entry["kind"] == "repair":
            entry.setdefault("family", "math")
            entry.setdefault("bench", "math")
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained("model-orig")
    math = {bench: [r for r in pool if r["family"] == "math" and r["bench"] == bench]
            for bench in ("gsm8k", "math")}
    lengths = {
        r["id"]: len(tok.encode(r["completion"] + "<|im_end|>", add_special_tokens=False, verbose=False))
        for r in itertools.chain.from_iterable(math.values())
    }
    draws = [(rng.sample(math["gsm8k"], 56), rng.sample(math["math"], 56)) for _ in range(64)]
    gsm, hard = min(draws, key=lambda pair: abs(sum(lengths[r["id"]] for r in pair[0])
                                                - sum(lengths[r["id"]] for r in pair[1])))
    train = gsm + hard
    for family, count in (("mc", 96), ("instruction", 80), ("logic", 32)):
        train += rng.sample([r for r in pool if r["family"] == family], count)
    rng.shuffle(train)

    exclusions = [
        "data/heldout.jsonl", "data/r2/mc-heldout.jsonl", "data/r2/math.jsonl",
        "data/r2/math-hints.jsonl", "data/r2/rehearsal.jsonl",
        "data/filter-candidates.jsonl", "data/commit.jsonl", "data/repair.jsonl",
    ]
    exclude_ids = {r["id"] for r in pool}
    exclude_prompts = {canonical(r["prompt"]) for r in pool}
    for path in exclusions:
        for item in rows(path):
            exclude_ids.add(item["id"])
            raw = item.get("question") or item.get("prompt_text")
            if raw:
                exclude_prompts.add(canonical(raw))
            if item.get("prompt"):
                exclude_prompts.add(canonical(item["prompt"]))
    dev = list(instructions())
    rng_dev = random.Random(SEED + 1)
    eligible = {name: [] for name in ("gsm8k", "math")}
    seen = set()
    for r in rows("data/train.jsonl"):
        text = canonical(r["question"])
        if r["id"] in exclude_ids or text in exclude_prompts or text in seen:
            continue
        seen.add(text)
        eligible[r["bench"]].append(r)
    for options in eligible.values():
        dev += rng_dev.sample(options, 32)
    assert len({r["id"] for r in train}) == len(train) == 320
    assert len({r["id"] for r in dev}) == len({canonical(r["question"]) for r in dev}) == 96
    assert not any(r["id"] in exclude_ids or canonical(r["question"]) in exclude_prompts for r in dev)
    rng_dev.shuffle(dev)
    out.mkdir()
    write(out / "train.jsonl", train)
    write(out / "dev.jsonl", dev)
    details = {
        "train_counts": dict(Counter(r["bench"] for r in train)),
        "dev_counts": dict(Counter(r["bench"] for r in dev)),
        "gsm8k_completion_tokens": sum(lengths[r["id"]] for r in gsm),
        "hard_math_completion_tokens": sum(lengths[r["id"]] for r in hard),
        "train_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((out / "dev.jsonl").read_bytes()).hexdigest(),
        "source_mix_sha256": MIX_HASH,
    }
    (out / "inputs.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details), flush=True)


def compare(baseline, candidate):
    base = {row["id"]: row for row in rows(baseline)}
    cand = {row["id"]: row for row in rows(candidate)}
    assert len(base) == 96 and base.keys() == cand.keys()
    result = {}
    for bench in ("gsm8k", "math", "instructions"):
        pairs = [(row, cand[key]) for key, row in base.items() if row["bench"] == bench]
        assert len(pairs) == 32
        assert all(a["bench"] == b["bench"] and a["seed"] == b["seed"] for a, b in pairs)
        wins = sum(not a["correct"] and b["correct"] for a, b in pairs)
        losses = sum(a["correct"] and not b["correct"] for a, b in pairs)
        n = wins + losses
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
        result[bench] = {
            "n": len(pairs), "baseline_correct": sum(a["correct"] for a, _ in pairs),
            "candidate_correct": sum(b["correct"] for _, b in pairs),
            "baseline_truncated": sum(a["finish_reason"] == "length" for a, _ in pairs),
            "candidate_truncated": sum(b["finish_reason"] == "length" for _, b in pairs),
            "wins": wins, "losses": losses, "paired_sign_p": p,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("out", type=Path)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("base")
    compare_parser.add_argument("cand")
    compare_parser.add_argument("aggregate", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    else:
        result = compare(args.base, args.cand)
        with args.aggregate.open("x") as out:
            json.dump(result, out, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
