"""Generate and select round-two thinking traces from training-split or original prompts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from statistics import median
from urllib.request import Request, urlopen

from grpo_gates import PROMPT, boxed, is_correct as math_correct
from think_probe import call

FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)
MIX = {"math": 7, "mc": 6, "instruction": 5, "logic": 2}
URL = "http://127.0.0.1:18000/v1"


def rows(path):
    with open(path) as source:
        return [json.loads(line) for line in source]


def write(path, entries):
    with open(path, "x") as dest:
        for item in entries:
            dest.write(json.dumps(item) + "\n")


def item_seed(item, sample):
    digest = hashlib.sha256(item["id"].encode()).digest()
    return (int.from_bytes(digest[:4]) + 20261126 + sample) % (2**31 - 1)


def prompt(item):
    return item["prompt_text"] if "prompt_text" in item else PROMPT.format(q=item["question"])


def family(item):
    return item.get("family", "math")


def answer(item, final):
    if family(item) == "math":
        return (boxed(final) or item["answer"]) if math_correct(item["bench"], final, item["answer"]) else None
    if family(item) == "mc":
        from r2_mc import choice
        selected = choice(item, final)
        return selected if selected == item["answer"] else None
    from r2_rehearsal import is_correct
    return "pass" if is_correct(item, final) else None


def completion(choice):
    message = choice["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning and "</think>" not in text:
        text = reasoning + ("\n</think>\n\n" + text if text else "")
    return text


def chat_head(tok, item):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt(item)}],
        tools=item.get("tools"), tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def completion_request(gen, seed, prompt_text, max_tokens, timeout):
    body = {
        "model": "policy", "prompt": prompt_text,
        "max_tokens": max_tokens, "seed": seed, "skip_special_tokens": False,
        **{key: gen[key] for key in ("temperature", "top_p", "top_k", "presence_penalty")},
    }
    req = Request(URL + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sample(item, seed, gen, tool_head):
    if tool_head is None:
        result = call(URL, "policy", prompt(item), seed, gen)
        choice, = result["choices"]
        text = completion(choice)
    else:
        result = completion_request(gen, seed, tool_head, gen["max_tokens"], 7200)
        choice, = result["choices"]
        text = choice["text"]
    return {
        "id": item["id"], "bench": item["bench"], "seed": seed,
        "text": text, "finish_reason": choice["finish_reason"],
        "completion_tokens": result["usage"]["completion_tokens"],
    }


def split_completion(text):
    thinking, separator, final = text.partition("</think>")
    if not separator or "</think>" in final or "<think>" in final:
        return thinking, ""
    return thinking, final.strip()


def canonical(text):
    return "".join(text.split()).casefold()


def validate_partition(held, training):
    def keys(group, key):
        return [key(item) for item in group]

    for label, key in (
        ("ID", lambda r: r["id"]),
        ("prompt", lambda r: canonical(prompt(r))),
    ):
        h, t = keys(held, key), keys(training, key)
        if len(h) != len(set(h)) or len(t) != len(set(t)):
            raise ValueError(f"Duplicate {label} in a split")
        if set(h).intersection(t):
            raise ValueError(f"Training/held-out {label} overlap")


def prep_math(args):
    held = rows(args.heldout)
    blocked = {canonical(prompt(item)) for item in held}
    by_bench = {"gsm8k": [], "math": []}
    for item in rows(args.source):
        question = canonical(prompt(item))
        if question not in blocked:
            blocked.add(question)
            by_bench[item["bench"]].append(item)
    rng = random.Random(20261126)
    selected = []
    for entries in by_bench.values():
        if len(entries) < args.per_bench:
            raise ValueError("Insufficient unique math prompts")
        selected.extend(rng.sample(entries, args.per_bench))
    validate_partition(held, selected)
    write(args.out, selected)
    print(f"Prepared {len(selected)} training-only math items")


def probe(args):
    items = rows(args.items)
    gen = json.loads(Path("evaluation.json").read_text())["generation"]
    tool_heads = {}
    if any(item.get("tools") for item in items):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("model-orig")
        tool_heads = {item["id"]: chat_head(tok, item) for item in items if item.get("tools")}
    jobs = []
    for index, item in enumerate(items):
        for n in range(args.samples):
            # Preserve round-one seeds for the existing 128 held-out math rows.
            seed = 20261124 + 1000 * index + n if args.paired else item_seed(item, n)
            jobs.append((item, seed, gen, tool_heads.get(item["id"])))
    with open(args.out, "x") as dest, ThreadPoolExecutor(args.conc) as executor:
        for i, result in enumerate(executor.map(lambda job: sample(*job), jobs), 1):
            dest.write(json.dumps(result) + "\n")
            if i % 50 == 0:
                dest.flush()
                print(f"sampled {i}/{len(jobs)}", flush=True)
    print(f"sampled {len(jobs)}/{len(jobs)}", flush=True)


def score(args):
    items = {item["id"]: item for item in rows(args.items)}
    seen = set()
    with open(args.out, "x") as dest:
        for row in rows(args.probe):
            item = items[row["id"]]
            if (row["id"], row["seed"]) in seen:
                raise ValueError("Duplicate rollout")
            seen.add((row["id"], row["seed"]))
            _, final = split_completion(row["text"])
            row["closed"] = "</think>" in row["text"]
            row["correct"] = bool(
                row["finish_reason"] == "stop" and final and answer(item, final) is not None
            )
            if not args.keep_text:
                row.pop("text")
            dest.write(json.dumps(row) + "\n")
    if {key[0] for key in seen} != items.keys():
        raise ValueError("Missing probe items")


def cut(text, fraction):
    offset = int(len(text) * fraction)
    line = text.rfind("\n", 0, offset)
    return line if line > 0 else offset


def force(head, prefix, seed, gen):
    result = completion_request(gen, seed, head + prefix + "\n</think>\n\n", 384, 3600)
    choice, = result["choices"]
    text = choice["text"]
    if choice["finish_reason"] == "stop" and "<think>" not in text and "</think>" not in text:
        return text.strip()
    return ""


def select(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("model-orig")
    items = {item["id"]: item for item in rows(args.items)}
    attempts = {key: [] for key in items}
    for probe_row in rows(args.scored):
        attempts[probe_row["id"]].append(probe_row)
    gen = json.loads(Path("evaluation.json").read_text())["generation"]

    def encoded(text):
        return len(tok.encode(text, add_special_tokens=False, verbose=False))

    def build(item, head, completion, kind):
        if encoded(head) + encoded(completion + "<|im_end|>") > 8192:
            return None
        return {
            "id": item["id"], "family": family(item), "bench": item["bench"],
            "kind": kind, "prompt": head, "completion": completion,
        }

    def pick(item):
        head = chat_head(tok, item)
        required = 50 if family(item) == "mc" else 1
        usable = []
        for row in attempts[item["id"]]:
            thinking, _ = split_completion(row["text"])
            if row["correct"] and encoded(thinking.strip()) >= required:
                example = build(item, head, row["text"], "finished")
                if example:
                    usable.append((row["completion_tokens"], example))
        if usable:
            middle = median(n for n, _ in usable)
            return min(usable, key=lambda pair: (abs(pair[0] - middle), -pair[0]))[1]
        if family(item) not in {"math", "mc"}:
            return None
        for row in attempts[item["id"]]:
            text = row["text"]
            if row["finish_reason"] != "length" or "</think>" in text:
                continue
            seen = set()
            previous_offset = 0
            for index, fraction in enumerate(FRACTIONS):
                offset = cut(text, fraction)
                prefix = text[:offset]
                if offset <= previous_offset or encoded(prefix.strip()) < required:
                    continue
                previous_offset = offset
                final = force(head, prefix, (row["seed"] + index + 2) % (2**31 - 1), gen)
                if family(item) == "math" and not boxed(final):
                    continue
                value = answer(item, final) if final else None
                if value is None:
                    continue
                if value in seen:
                    example = build(item, head, prefix + "\n</think>\n\n" + final, "commit")
                    if example:
                        return example
                seen.add(value)
        return None

    counts = Counter()
    prefix = rows(args.selected_prefix) if args.selected_prefix else []
    positions = {item_id: i for i, item_id in enumerate(items, 1)}
    prefix_positions = [positions[row["id"]] for row in prefix]
    if prefix_positions != sorted(set(prefix_positions)):
        raise ValueError("Selection checkpoint IDs must follow the original item order")
    offset = prefix_positions[-1] if prefix_positions else 0
    remaining = list(items.values())[offset:]
    with open(args.out, "x") as output, ThreadPoolExecutor(args.conc) as executor:
        for row in prefix:
            counts[f"{row['family']}/{row['kind']}"] += 1
            output.write(json.dumps(row) + "\n")
        output.flush()
        for i, example in enumerate(executor.map(pick, remaining), offset + 1):
            if example:
                counts[f"{example['family']}/{example['kind']}"] += 1
                output.write(json.dumps(example) + "\n")
            if i % 50 == 0:
                output.flush()
                print(f"selected {i}/{len(items)} {dict(counts)}", flush=True)
    print(json.dumps(counts), flush=True)


def mix(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("model-orig")
    held_ids = {row["id"] for path in args.holdout for row in rows(path)}
    pools = {family: {} for family in MIX}
    counts = Counter()
    repair = []
    for path in args.inputs:
        for row in rows(path):
            if row["id"] in held_ids:
                raise ValueError("Held-out example in training trace")
            size = len(tok.encode(row["prompt"], add_special_tokens=False, verbose=False))
            size += len(tok.encode(row["completion"] + "<|im_end|>", add_special_tokens=False, verbose=False))
            if size > 8192:
                continue
            key = family(row)
            if row["kind"] == "repair":
                repair.append(row)
            elif row["id"] not in pools[key]:
                pools[key][row["id"]] = row
    rng = random.Random(20261126)
    rng.shuffle(repair)
    for row in repair:
        if row["id"] not in pools["math"] or pools["math"][row["id"]]["kind"] != "finished":
            pools["math"][row["id"]] = row
    capacities = {key: len(pool) for key, pool in pools.items()}
    units = min(args.target // 20, *(capacities[key] // MIX[key] for key in MIX))
    print(json.dumps({"eligible": capacities, "balanced_size": units * 20}), flush=True)
    if units * 20 < args.minimum:
        raise ValueError(f"Balanced mix cannot reach {args.minimum} examples")
    mixed = []
    for key, pool in pools.items():
        candidates = list(pool.values())
        rng.shuffle(candidates)
        if key == "math":
            # Reserve some math slots for each available repair/commit mode.
            groups = {kind: [row for row in candidates if row["kind"] == kind] for kind in ("repair", "commit")}
            selected = [row for group in groups.values() for row in group[: units // 2]]
            used = {row["id"] for row in selected}
            candidates = selected + [row for row in candidates if row["id"] not in used]
        mixed.extend(candidates[: units * MIX[key]])
    rng.shuffle(mixed)
    for row in mixed:
        counts[f"{family(row)}/{row['kind']}"] += 1
    write(args.out, mixed)
    print(json.dumps(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("math")
    p.add_argument("source")
    p.add_argument("heldout")
    p.add_argument("out")
    p.add_argument("--per-bench", type=int, default=700)
    p.set_defaults(run=prep_math)
    p = sub.add_parser("probe")
    p.add_argument("items")
    p.add_argument("out")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--conc", type=int, default=128)
    p.add_argument("--paired", action="store_true")
    p.set_defaults(run=probe)
    p = sub.add_parser("score")
    p.add_argument("items")
    p.add_argument("probe")
    p.add_argument("out")
    p.add_argument("--keep-text", action="store_true")
    p.set_defaults(run=score)
    p = sub.add_parser("select")
    p.add_argument("items")
    p.add_argument("scored")
    p.add_argument("out")
    p.add_argument("--selected-prefix", help="Reuse an ordered partial selection from this data")
    p.add_argument("--conc", type=int, default=64)
    p.set_defaults(run=select)
    p = sub.add_parser("mix")
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--minimum", type=int, default=1500)
    p.set_defaults(run=mix)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
