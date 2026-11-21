#!/usr/bin/env python3
"""What a model did on "Evidence verdicts", beyond one score. Standard library only.

    python breakdown.py REPORT.json [--set FILE] [--save ANSWERS.json] [--json]
    python breakdown.py ANSWERS.json [MORE.json ...] [--versus OTHER.json [MORE.json ...]] [--json]
    python breakdown.py --rule [kinds|words] [--json]

REPORT.json is a report written by

    delta evaluate --validation evidence-verdicts-v1.json --output REPORT.json

and its generations file (REPORT.generations.jsonl) is read from beside it. --set names the item file
that run was given, when it is not the one in this folder; its sha256 must be the one in the report.
--save writes the run as a small per-item file (the letter read, the score, the token count; no
generated text), which this script reads back the same way. Items are matched by the sha256 of their
prompt, so a run made on another build of the set, or on a file of some of its items, lines up with this
one; several saved files are read as parts of one run (the same model and settings on different items).

For every item it reads the model's answer the way `delta evaluate` scores a custom file, turns the
letter back into a label through the item's own option order, and prints gold against answer:

  rules without evidence   on "not enough evidence" items, the model answered supported or refuted.
                           For a court this is the costly error: a ruling the evidence does not carry.
                           Reported apart for unrelated passages and for close ones (two or more content
                           words in common with the claim, none of its values).
  declines a clear case    on supported and refuted items, the model answered "not enough evidence".
  flips                    supported answered as refuted, or refuted as supported.
  no answer                no letter could be read: the answer was cut off at the token limit, the
                           thinking never closed, or the last line is not a boxed letter.

Items are not independent: some share a claim, and some show a passage another item shows too. Every
interval resamples groups (items linked by a shared claim, by passages from the same record, or by a
passage both prompts show word for word, transitively), and every sign test counts one vote per group;
--versus also gives the sign tests with one vote per claim. Output tokens are counted on this build's items only.

--versus compares the run with another, item by item. A run that leaves an answer unfinished neither scores
nor rules on that item, so the comparison also gives the score each run would have at its own accuracy in
each family had it answered its unfinished items, and the same comparison on the items both answered.

--rule scores a rule that uses no model, as a run, to show how far surface cues go:
  words   no content word in common with the passages: "not enough evidence"; a number or year of the
          claim missing from the passages: "refuted"; otherwise "supported".
  kinds   the same, except that a missing value means "refuted" only when the passages hold another
          value of the same kind (a year, a percentage, a number with a scale word, a plain number), and
          "not enough evidence" otherwise. This is the stronger of the two.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

NAME = "evidence-verdicts-v1"
FAMILIES = ["ev_supported", "ev_refuted", "ev_insufficient"]
LABELS = ["supported", "refuted", "insufficient"]
LETTERS = ["A", "B", "C"]
CLOSE = ("near_miss", "related")  # the strata of the third family whose passage is close to the claim
KEPT = ["model_id", "config_id", "source_version_id", "source_content_hash", "version_id", "artifact", "runtime", "generation",
        "generation_overridden", "evaluation_sha256", "validation_sha256", "validation_kind", "validation_score",
        "validation_benchmarks", "output_tokens", "generation_seconds", "truncation_rate", "evaluator", "text_engine",
        "text_engine_version", "task_protocol", "complete", "full_validation_set", "selected_benchmarks",
        "examples_per_benchmark_limit"]


def final_text(text, thinking):
    """The text `delta evaluate` scores for a custom file (delta_cli/public_eval/custom.py): with thinking
    on, what follows the last </think>, and nothing when the thinking never closed; with thinking off, all of it."""
    text = "" if text is None else str(text)
    if thinking:
        return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else ""
    return text.strip()


def read_choice(answer):
    """The letter the `choice` scorer compares with the gold answer (delta_cli/public_eval/custom_metrics.py)."""
    boxed = re.findall(r"\\boxed\{([A-E1-5])\}", answer)
    matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer)
    return matches[-1] if matches else None


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def answers_of_report(report_path, set_path):
    """A `delta evaluate` report and its generations, as per-item rows keyed by the sha256 of the prompt."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    raw = Path(set_path).read_bytes()
    if sha256(raw) != report.get("validation_sha256"):
        sys.exit(f"{report_path} was made on another item file (sha256 {str(report.get('validation_sha256'))[:12]}); "
                 f"name the file that run was given with --set")
    prompt_sha = {(fam, x["id"]): sha256(x["prompt"]) for fam, rows in json.loads(raw).items() for x in rows}
    gen_path = Path(report_path).with_suffix(".generations.jsonl")
    thinking = bool((report.get("generation") or {}).get("enable_thinking"))
    rows = []
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"benchmark": r["benchmark"], "prompt_sha256": prompt_sha[(r["benchmark"], r["example_id"])], "sample": r.get("sample", 0),
                     "letter": read_choice(final_text(r.get("text"), thinking)), "score": r.get("score"),
                     "tokens": r.get("tokens"), "truncated": bool(r.get("truncated"))})
    return {"name": NAME, "made_by": "breakdown.py --save, from a `delta evaluate --validation` report and its generations file",
            "run": {k: report[k] for k in KEPT if k in report}, "answers": rows}


def passages_of(prompt):
    """The passages a prompt shows, in order: the quoted text of each `[n] host: "..."` line."""
    out = []
    for line in prompt.split("\n")[4:]:
        found = re.fullmatch(r'\[\d+\] (\S+): "(.*)"', line)
        if not found:
            break
        out.append(found.group(2))
    return out


def groups_of(prov, shown=None):
    """Item id to group: items linked by a shared claim, by passages from the same record, or by a passage both
    prompts show word for word, transitively. `shown` maps an item id to the passages its prompt shows.
    A group is named by its smallest record id."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for p in prov.values():
        join(p["cluster"], p.get("evidence_from") or p["cluster"])
    first = {}
    for i, texts in sorted((shown or {}).items()):
        if i in prov:
            for text in texts:
                join(first.setdefault(text, prov[i]["cluster"]), prov[i]["cluster"])
    return {i: find(p["cluster"]) for i, p in prov.items()}


def tally(answers, prov, shown=None):
    """One record per answer whose item is in this build: family, claim, group, gold label, answered label or None.
    `shown` (item id to the passages its prompt shows) puts items that show the same passage in one group."""
    by_sha = {p["prompt_sha256"]: (i, p) for i, p in prov.items()}
    group = groups_of(prov, shown)
    out, absent, differs = [], 0, 0
    for a in answers:
        hit = by_sha.get(a["prompt_sha256"])
        if hit is None:
            absent += 1
            continue
        item_id, p = hit
        letter = a.get("letter")
        said = p["options"][LETTERS.index(letter)] if letter in LETTERS else None
        right = said == p["label"]
        if a.get("score") is not None and bool(a["score"]) != right:
            differs += 1
        out.append({"id": item_id, "family": "ev_" + p["family"], "cluster": p["cluster"], "group": group[item_id], "gold": p["label"],
                    "said": said, "right": right, "truncated": bool(a.get("truncated")), "tokens": a.get("tokens"), "stratum": p["stratum"]})
    return out, absent, differs


def group_of(x):
    return x.get("group", x["cluster"])


def macro(records):
    accs = []
    for fam in FAMILIES:
        xs = [x for x in records if x["family"] == fam]
        if xs:
            accs.append(sum(x["right"] for x in xs) / len(xs))
    return 100 * sum(accs) / len(accs) if accs else float("nan")


def percentile_interval(stats):
    stats = sorted(s for s in stats if s == s)
    if not stats:
        return None
    return [round(stats[int(0.025 * len(stats))], 1), round(stats[min(len(stats) - 1, int(0.975 * len(stats)))], 1)]


def bootstrap(units, statistic, draws=2000, seed=20261120):
    """A 95% percentile interval for `statistic` over rows, resampling groups with replacement.
    `units` maps a group to its rows."""
    keys, rng = sorted(units), random.Random(seed)
    return percentile_interval(statistic([row for k in rng.choices(keys, k=len(keys)) for row in units[k]]) for _ in range(draws))


def summarise(records):
    n = len(records)
    rate = lambda k, xs: round(100 * k / len(xs), 1) if xs else None
    fam = {}
    for f in FAMILIES:
        xs = [x for x in records if x["family"] == f]
        fam[f] = {"items": len(xs), "right": sum(x["right"] for x in xs), "accuracy": rate(sum(x["right"] for x in xs), xs)}
    matrix = {g: {s: 0 for s in LABELS + ["no answer"]} for g in LABELS}
    for x in records:
        matrix[x["gold"]][x["said"] or "no answer"] += 1
    ins = [x for x in records if x["gold"] == "insufficient"]
    clear = [x for x in records if x["gold"] in ("supported", "refuted")]
    none = [x for x in records if x["said"] is None]
    units = {}
    for x in records:
        units.setdefault(group_of(x), []).append(x)
    strata = {}
    for x in records:
        s = strata.setdefault(f"{x['family'].replace('ev_', '')}/{x['stratum']}", {"items": 0, "right": 0})
        s["items"] += 1
        s["right"] += int(x["right"])

    def ruling(xs):
        k = sum(x["said"] in ("supported", "refuted") for x in xs)
        return {"count": k, "of": len(xs), "rate": rate(k, xs)}
    declined = sum(x["said"] == "insufficient" for x in clear)
    flipped = sum(x["said"] in ("supported", "refuted") and x["said"] != x["gold"] for x in clear)
    return {
        "items": n, "source_sentences": len({x["cluster"] for x in records}), "groups": len(units),
        "output_tokens": sum(x["tokens"] or 0 for x in records),
        "score": round(macro(records), 1), "score_interval_95": bootstrap(units, macro) if records else None,
        "families": fam, "by_stratum": dict(sorted(strata.items())), "gold_by_answer": matrix,
        "rules_without_evidence": ruling(ins),
        "rules_without_evidence_on_unrelated_passages": ruling([x for x in ins if x["stratum"] not in CLOSE]),
        "rules_without_evidence_on_close_passages": ruling([x for x in ins if x["stratum"] in CLOSE]),
        "declines_a_clear_case": {"count": declined, "of": len(clear), "rate": rate(declined, clear)},
        "flips": {"count": flipped, "of": len(clear), "rate": rate(flipped, clear)},
        "no_answer": {"count": len(none), "of": n, "rate": rate(len(none), records),
                      "cut_off_at_the_token_limit": sum(x["truncated"] for x in none),
                      "not_cut_off": sum(not x["truncated"] for x in none)},
    }


def value_kind(t):
    return "year" if t["kind"] == "year" else "percent" if t["unit"] == "%" else "scaled" if t["scale"] > 0 else "plain"


def rule_answers(items, rule="kinds"):
    """What a rule with no model answers, as a run (see --rule above)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_set
    rows = []
    for family, examples in items.items():
        for x in examples:
            lines = x["prompt"].split("\n")
            title, claim, passages = lines[0].split('"')[1], lines[1], passages_of(x["prompt"])
            options = [line[3:].split(":")[0] for line in lines if re.match(r"[ABC]\. ", line)]
            title_words = build_set.title_words_of(title)
            in_passages = {w for p in passages for w in build_set.content_words(p, title_words)}
            held = set().union(*(build_set.value_set(p) for p in passages))
            claim_values = build_set.value_set(claim)
            other_kinds = {value_kind(t) for p in passages for t in build_set.number_tokens(p)
                           if t["value"] is not None and f"{t['value']}|{t['unit']}" not in claim_values}
            missing = [t for t in build_set.number_tokens(claim) if t["value"] is not None and f"{t['value']}|{t['unit']}" not in held]
            if not any(w in in_passages for w in build_set.content_words(claim, title_words)):
                said = "Not enough evidence"
            elif not missing:
                said = "Supported"
            elif rule == "words" or any(value_kind(t) in other_kinds for t in missing):
                said = "Refuted"
            else:
                said = "Not enough evidence"
            letter = LETTERS[options.index(said)]
            rows.append({"benchmark": family, "prompt_sha256": sha256(x["prompt"]), "sample": 0, "letter": letter,
                         "score": float(letter == x["answer"]), "tokens": 0, "truncated": False})
    names = {"words": "a word-and-number rule, no model", "kinds": "a word-and-number rule that compares values of the same kind, no model"}
    return {"name": NAME, "run": {"model_id": names[rule]}, "answers": rows, "parts": []}


SAME = ["model_id", "source_version_id", "artifact", "generation"]


def load_run(paths, set_path):
    """One run from one or more files. Several files are parts of one run: the same model and settings on different items."""
    parts = []
    for path in paths:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        parts.append(loaded if "answers" in loaded else answers_of_report(path, set_path))
    first = parts[0].get("run") or {}
    seen, answers = set(), []
    for part in parts:
        run = part.get("run") or {}
        if any(run.get(k) != first.get(k) for k in SAME):
            sys.exit("the files are not parts of one run: the model or the generation settings differ")
        for row in part["answers"]:
            key = (row["prompt_sha256"], row.get("sample", 0))
            if key in seen:
                sys.exit("the files are not parts of one run: an item is answered in two of them")
            seen.add(key)
            answers.append(row)
    return {**parts[0], "answers": answers,
            "parts": [{"validation_sha256": (p.get("run") or {}).get("validation_sha256"), "answers": len(p["answers"])} for p in parts]}


def sign_test(a, b):
    """Two-sided exact sign test on a and b votes."""
    n = a + b
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ruled(x):
    return x["said"] in ("supported", "refuted")


def paired_rulings(pairs):
    """On "not enough evidence" items: how often each run ruled anyway, the items only one of them ruled on,
    and an exact sign test that counts one vote per group (the direction of that group's items, ties left out)."""
    ins = [(x, b) for x, b in pairs if x["gold"] == "insufficient"]
    here, there = sum(ruled(x) and not ruled(b) for x, b in ins), sum(ruled(b) and not ruled(x) for x, b in ins)
    def votes(unit):
        lean = {}
        for x, b in ins:
            lean[unit(x)] = lean.get(unit(x), 0) + (ruled(x) and not ruled(b)) - (ruled(b) and not ruled(x))
        return len(lean), [sum(v > 0 for v in lean.values()), sum(v < 0 for v in lean.values())]
    groups, by_group = votes(group_of)
    _, by_claim = votes(lambda x: x["cluster"])
    return {"this_run": sum(ruled(x) for x, _ in ins), "other_run": sum(ruled(b) for _, b in ins), "of": len(ins), "groups": groups,
            "only_this_run": here, "only_other_run": there, "group_votes": by_group, "sign_test_p": round(sign_test(*by_group), 3),
            "claim_votes": by_claim, "sign_test_p_by_claim": round(sign_test(*by_claim), 3)}


def paired_scores(pairs):
    """The difference in score on the given pairs, with a 95% interval that resamples groups."""
    units = {}
    for x, b in pairs:
        units.setdefault(group_of(x), []).append((x, b))
    mine, base = [x for x, _ in pairs], [b for _, b in pairs]
    families = {}
    for fam in FAMILIES:
        xs = [(x, b) for x, b in pairs if x["family"] == fam]
        if xs:
            families[fam] = round(100 * (sum(x["right"] for x, _ in xs) - sum(b["right"] for _, b in xs)) / len(xs), 1)
    return {"items": len(pairs), "score": round(macro(mine), 1), "other_score": round(macro(base), 1),
            "difference": round(macro(mine) - macro(base), 1),
            "difference_interval_95": bootstrap(units, lambda rows: macro([x for x, _ in rows]) - macro([b for _, b in rows])) if pairs else None,
            "difference_by_family": families}


def at_own_rate(rows):
    """The score a run would have had it answered its unfinished items at its own accuracy in each family."""
    accs = []
    for fam in FAMILIES:
        xs = [x for x in rows if x["family"] == fam]
        if not xs:
            continue
        done = [x for x in xs if x["said"] is not None]
        right = sum(x["right"] for x in done)
        accs.append((right + (right / len(done) if done else 0.0) * (len(xs) - len(done))) / len(xs))
    return 100 * sum(accs) / len(accs) if accs else float("nan")


def versus(records, other):
    """This run against another, item by item, on every item both runs have."""
    theirs = {x["id"]: x for x in other}
    pairs = [(x, theirs[x["id"]]) for x in records if x["id"] in theirs]
    both = [(x, b) for x, b in pairs if x["said"] is not None and b["said"] is not None]
    clear = [(x, b) for x, b in both if x["gold"] != "insufficient"]
    left = [x for x, b in pairs if b["said"] is None]  # this run's answers on the items the other run left unfinished
    rest = [x for x, b in pairs if b["said"] is not None]
    mine, base = [x for x, _ in pairs], [b for _, b in pairs]
    ins_base = [b for _, b in pairs if b["gold"] == "insufficient"]
    finished = [b for b in ins_base if b["said"] is not None]
    return {"items_in_both_runs": len(pairs), "groups": len({group_of(x) for x, _ in pairs}), **paired_scores(pairs),
            "no_answer": {"this_run": sum(x["said"] is None for x in mine), "other_run": sum(b["said"] is None for b in base), "of": len(pairs)},
            "at_own_rate": {"score": round(at_own_rate(mine), 1), "other_score": round(at_own_rate(base), 1),
                            "difference": round(at_own_rate(mine) - at_own_rate(base), 1)},
            "items_the_other_run_left_unfinished": {"items": len(left), "this_run_right": sum(x["right"] for x in left),
                                                    "rest": len(rest), "this_run_right_on_the_rest": sum(x["right"] for x in rest)},
            "rules_without_evidence": {**paired_rulings(pairs), "other_run_unfinished": len(ins_base) - len(finished),
                                       "other_run_at_own_rate": round(sum(ruled(b) for b in finished) * len(ins_base) / len(finished), 1) if finished else None},
            "rules_without_evidence_on_close_passages": paired_rulings([(x, b) for x, b in pairs if x["stratum"] in CLOSE]),
            "when_both_gave_a_letter": {**paired_scores(both), "rules_without_evidence": paired_rulings(both),
                                        "declines_a_clear_case": {"this_run": sum(x["said"] == "insufficient" for x, _ in clear),
                                                                  "other_run": sum(b["said"] == "insufficient" for _, b in clear), "of": len(clear)}}}


def main():
    ap = argparse.ArgumentParser(description="Gold against answer for one run on Evidence verdicts.")
    ap.add_argument("run", nargs="*", help="a `delta evaluate --validation` report, or files written by --save (several files are parts of one run)")
    ap.add_argument("--rule", nargs="?", const="kinds", choices=["kinds", "words"], help="instead of a run, score a rule that uses no model")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent), help="folder with evidence-verdicts-v1.json and provenance.json")
    ap.add_argument("--set", help="the item file the run was given, when it is not the one in --dir")
    ap.add_argument("--save", help="write the run as a small per-item file with no generated text")
    ap.add_argument("--versus", nargs="+", help="another run (files written by --save) to compare this one with, item by item")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    folder = Path(a.dir)
    set_path = folder / f"{NAME}.json"
    if bool(a.rule) == bool(a.run):
        sys.exit("give one run (a report, or files written by --save), or --rule")
    data = rule_answers(json.loads(set_path.read_text(encoding="utf-8")), a.rule) if a.rule else load_run(a.run, a.set or set_path)
    if a.save:
        if len(a.run) > 1:
            sys.exit("--save takes one report")
        Path(a.save).write_text(json.dumps({k: v for k, v in data.items() if k != "parts"}, indent=1) + "\n", encoding="utf-8", newline="\n")
    shown = {x["id"]: passages_of(x["prompt"]) for rows in json.loads(set_path.read_text(encoding="utf-8")).values() for x in rows}
    prov = {i: p for i, p in json.loads((folder / "provenance.json").read_text(encoding="utf-8")).items() if i in shown}
    records, absent, differs = tally(data["answers"], prov, shown)
    s = summarise(records)
    run = data.get("run") or {}
    s["run"] = {"model_id": run.get("model_id"), "source_version_id": run.get("source_version_id"), "artifact": run.get("artifact"),
                "generation": run.get("generation"), "parts": data["parts"],
                "set_sha256_here": sha256(set_path.read_bytes()), "answers_for_items_not_in_this_build": absent,
                "answers_where_the_letter_read_here_disagrees_with_the_saved_score": differs}
    if a.versus:
        other, _, _ = tally(load_run(a.versus, set_path)["answers"], prov, shown)
        s["versus"] = versus(records, other)
    if a.json:
        print(json.dumps(s, indent=1))
        return
    g = run.get("generation") or {}
    head = run.get("model_id") if a.rule else f"thinking {'on' if g.get('enable_thinking') else 'off'}, up to {g.get('max_tokens')} tokens"
    print(f"{NAME}: {s['items']} items from {s['source_sentences']} source sentences in {s['groups']} groups; {head}")
    if absent:
        print(f"note: {absent} answers are for items that are not in this build of the set, and are left out")
    if differs:
        print(f"note: on {differs} answers the letter read here disagrees with the score saved in the run")
    lo, hi = s["score_interval_95"] or [float("nan")] * 2
    print(f"score (mean of the three family accuracies): {s['score']}   95% interval by group: {lo} to {hi}")
    for f in FAMILIES:
        v = s["families"][f]
        print(f"  {f:<16} {v['right']:>3} of {v['items']:<3} right" + (f"  ({v['accuracy']}%)" if v["accuracy"] is not None else ""))
    print("  by stratum: " + ", ".join(f"{k} {v['right']}/{v['items']}" for k, v in s["by_stratum"].items()))
    print("gold \\ answer      supported  refuted  insufficient  no answer")
    for gold in LABELS:
        m = s["gold_by_answer"][gold]
        print(f"  {gold:<14} {m['supported']:>9} {m['refuted']:>8} {m['insufficient']:>13} {m['no answer']:>10}")
    for key, words in (("rules_without_evidence", "rules without evidence"),
                       ("rules_without_evidence_on_unrelated_passages", "  when the passage is unrelated"),
                       ("rules_without_evidence_on_close_passages", "  when the passage is close to the claim"),
                       ("declines_a_clear_case", "declines a clear case"), ("flips", "flips supported and refuted")):
        v = s[key]
        print(f"{words}: {v['count']} of {v['of']}" + (f" ({v['rate']}%)" if v["rate"] is not None else ""))
    v = s["no_answer"]
    print(f"no answer: {v['count']} of {v['of']} ({v['cut_off_at_the_token_limit']} cut off at the token limit, {v['not_cut_off']} not cut off)")
    if "versus" in s:
        v = s["versus"]
        lo, hi = v["difference_interval_95"]
        print(f"against the other run, on the {v['items_in_both_runs']} items in both runs ({v['groups']} groups): {v['score']} against "
              f"{v['other_score']}, a difference of {v['difference']:+} (95% interval by group: {lo:+} to {hi:+})")
        print("  by family: " + ", ".join(f"{fam} {d:+}" for fam, d in v["difference_by_family"].items()))
        n, o = v["no_answer"], v["at_own_rate"]
        print(f"  no answer: {n['this_run']} in this run, {n['other_run']} in the other; at each run's own rate on those: "
              f"{o['score']} against {o['other_score']}, a difference of {o['difference']:+}")
        u = v["items_the_other_run_left_unfinished"]
        if u["items"]:
            print(f"  this run gets {u['this_run_right']} of the {u['items']} items the other left unfinished right, "
                  f"and {u['this_run_right_on_the_rest']} of the other {u['rest']}")
        r = v["rules_without_evidence"]
        print(f"  rules without evidence: {r['this_run']} against {r['other_run']} of {r['of']} (the other left {r['other_run_unfinished']} "
              f"unfinished; at its own rate, {r['other_run_at_own_rate']}); only this run on {r['only_this_run']} items, only the other on "
              f"{r['only_other_run']}; by group {r['group_votes'][0]} against {r['group_votes'][1]} (exact sign test p = {r['sign_test_p']})")
        w = v["when_both_gave_a_letter"]
        r = w["rules_without_evidence"]
        print(f"  on the {w['items']} items both answered (a subset the unfinished answers select): {w['score']} against {w['other_score']}; "
              f"rules without evidence {r['this_run']} against {r['other_run']} of {r['of']}, by group {r['group_votes'][0]} against "
              f"{r['group_votes'][1]} (p = {r['sign_test_p']})")


if __name__ == "__main__":
    main()
