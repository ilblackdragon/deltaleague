#!/usr/bin/env python3
"""The blind human read, in a terminal. Standard library only.

    python hand_check.py --reader NAME [--per 6] [--dir DIR]
    python hand_check.py --list [--per 6]

One sample, read in one sitting, one item at a time, exactly as a model sees it, with no label, family or
model answer, and nothing revealed afterwards. The sample is fixed before anyone reads:

  - PER items of each of four parts: supported; refuted; "not enough evidence" with an unrelated passage;
    and "not enough evidence" with a close passage (near_miss or related). In each part, the items with the
    lowest sha256("evidence-verdicts-v1|handcheck|" + key) among those that passed the solver gate;
  - every flagged item: one that a hosted model outside the gate answered differently, or that flagged.json
    names (an item a second reader found arguable). A second reader cannot remove an item; a person can.

All are shown mixed, in the order of sha256("evidence-verdicts-v1|handcheck-order|" + key), so neither the
family nor the reason an item is there can be read off the sequence.

The rule the reader goes by, shown before the first item:
  Supported: the evidence shows every part of the claim is true. Refuted: it shows some part of the claim is
  false. Not enough evidence: anything else, including evidence that shows only part of the claim, and
  evidence that could only be connected to the claim with something it does not say. Working a number out
  from what the evidence states is fine.

Answers go to handcheck.json after every item, with the sha256 of the prompt read, so a reader can stop and
come back. build_set.py then removes every item a reader answered differently from its label or could not
decide, while its prompt is still the one read. Nothing is ever relabelled, and a second reader can only
remove more.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_set  # noqa: E402

NOTE = ("The blind human read: one row per item read, with the sha256 of the prompt it was read on. `agrees` is whether the "
        "reader's letter is the item's gold letter; `flag` is set when the reader could not decide. Either removes the item "
        "from the final set, while the item's prompt is still the one read.")
RULE = ("Supported: the evidence shows every part of the claim is true. Refuted: it shows some part of the claim is false. "
        "Not enough evidence: anything else, including evidence that shows only part of the claim, and evidence you could "
        "only connect to the claim with something it does not say. Working a number out from what it states is fine.")
PARTS = [("supported", lambda x: x["family"] == "ev_supported"), ("refuted", lambda x: x["family"] == "ev_refuted"),
         ("unrelated", lambda x: x["family"] == "ev_insufficient" and x["stratum"] not in build_set.HARD),
         ("close", lambda x: x["family"] == "ev_insufficient" and x["stratum"] in build_set.HARD)]


def load(folder, name, default=None):
    path = Path(folder) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def pool_of(folder):
    """The items that passed the solver gate."""
    cfg = load(folder, "config.json")
    items, _, _ = build_set.candidates(load(folder, "records.json")["records"], cfg)
    passed, _ = build_set.gate_passed(items, cfg, load(folder, "solvers.json", {"models": {}}))
    return passed


def flagged_keys(folder, pool):
    """Items a hosted model outside the gate answered differently, or with no readable letter, and items flagged.json names."""
    gate = load(folder, "config.json")["gate_models"]
    models = load(folder, "solvers.json", {"models": {}}).get("models", {})
    listed = {r["key"] for r in load(folder, "flagged.json", {"rows": []})["rows"]}
    out = set()
    for item in pool:
        answers = [a[item["sha"]]["letter"] for name, a in sorted(models.items()) if name not in gate and item["sha"] in a]
        if item["key"] in listed or any(letter != item["answer"] for letter in answers):
            out.add(item["key"])
    return out


def part_of(item, flagged):
    return "flagged" if item["key"] in flagged else next(name for name, inside in PARTS if inside(item))


def sample(pool, per, flagged=frozenset()):
    """Which items are read, and in what order."""
    picked = [x for x in pool if x["key"] in flagged]
    for _, inside in PARTS:
        rows = [x for x in pool if inside(x) and x["key"] not in flagged]
        picked += build_set.by_key(rows, lambda x: build_set.order_key("handcheck", x["key"]))[:per]
    return build_set.by_key(picked, lambda x: build_set.order_key("handcheck-order", x["key"]))


def row_of(item, reader, answer, part, why="", seconds=None):
    letter = None if answer == "?" else answer
    return {"key": item["key"], "reader": reader, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "part": part,
            "prompt_sha256": item["sha"], "letter": letter, "label": item["options"][build_set.LETTERS.index(letter)] if letter else None,
            "agrees": letter == item["answer"] if letter else None,
            "flag": (why or "the reader could not decide") if answer == "?" else None, "seconds": seconds}


def main(argv=None, ask=input, say=print):
    ap = argparse.ArgumentParser(description="Read a fixed sample of the set blind and record the answers.")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--reader", default="r2", help="a short name for the reader; v1's own reader is r1")
    ap.add_argument("--per", type=int, default=6, help="items of each part")
    ap.add_argument("--list", action="store_true", help="only print which items are in the sample")
    a = ap.parse_args(argv)
    pool = pool_of(a.dir)
    flagged = flagged_keys(a.dir, pool)
    picked = sample(pool, a.per, flagged)
    if a.list:
        for item in picked:
            say(f"{item['key']}  ({part_of(item, flagged)})")
        return 0
    path = Path(a.dir) / "handcheck.json"
    store = load(a.dir, "handcheck.json", {})
    store = {"name": build_set.NS, **store, "note": NOTE, "rows": list(store.get("rows", []))}
    done = {r["key"] for r in store["rows"] if r["reader"] == a.reader and r.get("prompt_sha256") in {x["sha"] for x in picked}}
    todo = [x for x in picked if x["key"] not in done]
    say(f"Blind read, reader {a.reader}: {len(todo)} of {len(picked)} items left. Go only by the evidence shown.")
    say(RULE)
    say("Type A, B or C and press Enter; ? if you cannot decide; q to stop.")
    for number, item in enumerate(todo, len(picked) - len(todo) + 1):
        say("\n" + "=" * 78 + f"\nItem {number} of {len(picked)}\n" + "=" * 78)
        say(item["prompt"][:item["prompt"].rindex("\n")] + "\n")  # without the instruction line written for models
        started, answer = time.time(), ""
        while answer not in ("A", "B", "C", "?", "Q"):
            answer = ask("Your answer (A, B, C, ? or q): ").strip().upper()
        if answer == "Q":
            break
        why = ask("In a few words, why is it unclear? ").strip() if answer == "?" else ""
        store["rows"] = [r for r in store["rows"] if not (r["reader"] == a.reader and r["key"] == item["key"])]
        store["rows"].append(row_of(item, a.reader, answer, part_of(item, flagged), why, round(time.time() - started)))
        path.write_text(json.dumps(store, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    mine = [r for r in store["rows"] if r["reader"] == a.reader]
    say(f"\nSaved. Reader {a.reader} has read {len(mine)} of {len(picked)} items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
