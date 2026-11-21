#!/usr/bin/env python3
"""Finds the close passage for a "not enough evidence" item. Standard library only.

    python near_miss.py RECORDS.json --pages DIR [--config config.json] [--out RECORDS.json]
    python near_miss.py RECORDS.json --pages DIR --check

DIR holds the saved text of each receipt's page as DIR/<receipt key>.txt. For every true record whose claim
states a quantity or a year, this fills the record's `near` field with one sentence of its own sources that
qualifies, or null; --check compares instead and exits 1 on a difference. build_set.py checks the rule again
on the text a prompt shows, so a hand-edited `near` that breaks it is vetoed.

A sentence qualifies when, as a prompt would show it:
  - it shares at least two content words with the claim (letters only, four or more of them, not a stop
    word, not a word of the article's title, not a scale or approximation word in close_ignore_words);
  - it names no one on the list of names in config.people;
  - it holds none of the claim's values, and none of the bare digits of the claim's numbers;
  - it is not inside a passage the claim's own item shows, and does not contain one.
Among the sentences that qualify, one that holds a decoy comes first: another value of the same kind as
one of the claim's (a year for a year, a percentage for a percentage, a number with a scale word for one,
a plain number for a plain one). Then the most shared words; on a tie, the earlier sentence, and across a
claim's sources, the earlier receipt. A sentence is cut at sentence ends by the rule in sentence_spans and
shown tidied (tidy); it holds at most `caps.excerpt_chars` characters.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_set as b  # noqa: E402

A = re.ASCII
ABBREV = {"u.s", "u.k", "e.g", "i.e", "etc", "vs", "no", "inc", "ltd", "co", "corp", "dr", "mr", "mrs", "ms", "st", "jr", "sr",
          "fig", "sec", "approx", "est", "u.s.a", "a.m", "p.m"}
OPENERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"'([“‘")
ENTITY = {"amp": 38, "lt": 60, "gt": 62, "quot": 34, "apos": 39, "nbsp": 160, "ndash": 0x2013, "mdash": 0x2014, "hellip": 0x2026,
          "lsquo": 0x2018, "rsquo": 0x2019, "ldquo": 0x201C, "rdquo": 0x201D, "copy": 0xA9, "reg": 0xAE, "trade": 0x2122, "euro": 0x20AC,
          "pound": 0xA3, "para": 0xB6, "middot": 0xB7, "laquo": 0xAB, "raquo": 0xBB, "deg": 0xB0, "times": 0xD7, "bull": 0x2022,
          "rarr": 0x2192, "larr": 0x2190}


def edge_after(t, i):
    """Is there a sentence end right after index i: a terminator, one space, then a capital, a digit or an opening mark?"""
    c = t[i]
    if c not in ".!?" or i + 1 >= len(t) or t[i + 1] != " ":
        return False
    nxt = t[i + 2] if i + 2 < len(t) else ""
    if nxt not in OPENERS or nxt == "":
        return False
    if c == ".":
        s = i
        while s > 0 and t[s - 1] != " ":
            s -= 1
        word = t[s:i].lower()
        if word in ABBREV or re.fullmatch("[a-z]", word, A) or (re.fullmatch("[0-9]+", word, A) and nxt.isascii() and nxt.isdigit()):
            return False  # "U.S. Treasury", "J. Smith", "1. 2."
    return True


def sentence_spans(text, cap=300):
    """Every sentence as a [start, end) span; one shorter than 25 or longer than `cap` characters is left out."""
    out, s = [], 0
    for i in range(len(text)):
        if i == len(text) - 1 or edge_after(text, i):
            e = i + 1
            if 25 <= e - s <= cap:
                out.append((s, e))
            s = e + 1
    return out


def from_code(m, cp):
    return chr(cp) if 0 < cp <= 0x10FFFF else m.group(0)


def tidy(s):
    """A passage as a prompt shows it: HTML entities the saved text kept become characters, the stray capital A
    with a circumflex that a no-break space read in the wrong encoding leaves is dropped, whitespace is single."""
    s = re.sub(r"&#(\d{1,7});", lambda m: from_code(m, int(m.group(1))), s, flags=A)
    s = re.sub(r"&#x([0-9a-fA-F]{1,6});", lambda m: from_code(m, int(m.group(1), 16)), s)
    s = re.sub(r"&([a-zA-Z]{2,6});", lambda m: chr(ENTITY[m.group(1).lower()]) if m.group(1).lower() in ENTITY else m.group(0), s)
    s = s.replace("Â ", " ")
    return re.sub(f"^{b.JS_SPACE}|{b.JS_SPACE}$", "", re.sub(b.JS_SPACE, " ", s))


def literal(s):
    """A pattern that matches the text literally (the receipts' patterns are regular expressions)."""
    return re.sub(r"[\\^$.|?*+()\[\]{}]", lambda m: "\\" + m.group(0), s)


def near_passage(text, claim, title, shown, cap=300, ignore=(), people=None):
    """The qualifying sentence of one saved page, or None. `text` has its whitespace collapsed to single spaces.
    `ignore` are words that do not count as shared (config.close_ignore_words); a sentence that `people` matches is skipped."""
    tw = b.title_words_of(title)
    words = {w for w in b.content_words(claim, tw) if w not in ignore}
    values, digits, kinds = b.value_set(claim), b.bare_digits(claim), b.value_kinds(claim)
    if not values:
        return None
    best = None
    for s, e in sentence_spans(text, cap):
        excerpt = tidy(text[s:e])
        if people is not None and people.search(excerpt):
            continue
        if values & b.value_set(excerpt):
            continue
        bare = re.sub(r"[^0-9]+", "", excerpt)
        if any(d in bare for d in digits):
            continue
        if any(x and (excerpt in x or x in excerpt) for x in shown):
            continue
        shared = sorted(w for w in b.content_words(excerpt, tw) if w in words)
        if len(shared) < 2:
            continue
        decoy = bool(b.value_kinds(excerpt) & kinds)
        if best is None or (decoy and not best["decoy"]) or (decoy == best["decoy"] and len(shared) > len(best["shared"])):
            pattern = re.sub(f"^{b.JS_SPACE}|{b.JS_SPACE}$", "", text[s:s + 60])
            best = {"excerpt": excerpt, "pattern": literal(pattern), "shared": shared, "decoy": decoy}
    return best


def near_for_record(record, page_text, cfg):
    """The `near` field of a record: the best qualifying sentence across its receipts, in the records.json shape.
    `page_text(key)` returns a receipt's saved text, or None when it is not at hand."""
    if record.get("verdict") != "current" or not b.value_set(record["claim"]):
        return None

    def excluded(x):
        host, url = x.get("host") or "", str(x.get("url"))
        return any(host == h or host.endswith("." + h) or f"/{h}/" in url or f"//{h}" in url or f".{h}/" in url for h in cfg["excluded_hosts"])
    shown = [x["excerpt"] for x in record["receipts"] if x.get("excerpt")]
    best = None
    for rc in record["receipts"]:
        if rc["kind"] not in cfg["kinds_in_prompts"] or excluded(rc) or not rc.get("excerpt") or rc.get("status") != 200 or not rc.get("sha256"):
            continue
        raw = page_text(rc["key"])
        if raw is None:
            continue
        hit = near_passage(re.sub(b.JS_SPACE, " ", raw), record["claim"], record["article"]["title"], shown, cfg["caps"]["excerpt_chars"],
                           ignore=set(cfg.get("close_ignore_words") or []), people=re.compile(cfg["people"], A))
        if hit and (best is None or (hit["decoy"] and not best[1]["decoy"]) or (hit["decoy"] == best[1]["decoy"] and len(hit["shared"]) > len(best[1]["shared"]))):
            best = (rc, hit)
    if best is None:
        return None
    rc, hit = best
    return {"key": rc["key"], "url": rc["url"], "host": rc["host"], "kind": rc["kind"], "fetched_at": rc["fetched_at"], "status": rc["status"],
            "sha256": rc["sha256"], "pattern": hit["pattern"], "excerpt": hit["excerpt"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill or check the `near` field of every record from saved page texts.")
    ap.add_argument("records", help="records.json")
    ap.add_argument("--pages", required=True, action="append", help="a folder of saved page texts, <receipt key>.txt; repeatable")
    ap.add_argument("--config", help="config.json, when it is not beside the records")
    ap.add_argument("--out", help="write the records with their near fields here")
    ap.add_argument("--check", action="store_true", help="compare with the near fields already in the records")
    a = ap.parse_args(argv)
    path = Path(a.records)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = json.loads(Path(a.config or path.with_name("config.json")).read_text(encoding="utf-8"))

    def page_text(key):
        for folder in a.pages:
            f = Path(folder) / f"{key}.txt"
            if f.exists():
                return f.read_bytes().decode("utf-8", errors="replace")
        return None
    differ = 0
    for r in data["records"]:
        near = near_for_record(r, page_text, cfg)
        if a.check:
            if near != r.get("near"):
                differ += 1
                print(f"{r['id']}: differs", file=sys.stderr)
        else:
            r["near"] = near
    if a.check:
        print(json.dumps({"records": len(data["records"]), "with_near": sum(1 for r in data["records"] if r.get("near")), "differ": differ}))
        return 1 if differ else 0
    if a.out:
        rows = ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in data["records"])
        head = "".join(f"{json.dumps(k)}:{json.dumps(v, ensure_ascii=False, separators=(',', ':'))}," for k, v in data.items() if k != "records")
        Path(a.out).write_text("{" + head + '"records":[\n' + rows + "\n]}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"records": len(data["records"]), "with_near": sum(1 for r in data["records"] if r.get("near"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
