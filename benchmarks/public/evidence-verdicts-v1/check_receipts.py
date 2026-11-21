#!/usr/bin/env python3
"""Fetches every source the set quotes and tests each shown passage on the page as it is today. Standard library only.

    python check_receipts.py [--dir DIR] [--limit N] [--out REPORT.json] [--strict]

A receipt is a source URL, the sha256 of what that URL returned when it was saved, and a pattern that
matched in the saved text. The passage a prompt shows was cut from that saved text: around the pattern's
match, or, for a true claim, around the sentence of the same page that shares the most words with it, or,
for a close passage, the sentence near_miss.py chose. This script does not need the saved copies. It asks
each https URL again and reports two things for every passage the set shows:

  the receipt today
    same bytes         the page returns exactly what was saved
    still matches      the page changed, and the receipt's pattern still matches in it
    no longer matches  the page changed and the pattern is gone (the source moved on, or blocks scripts)
    a PDF, not read    the source is a PDF; it was read as extracted text, which this script cannot make
    not fetched        no answer, an HTTP error, or not an https URL
  the passage today
    on the page        the shown passage, as a prompt shows it, is in today's page text
    not on the page    it is not (the page changed there, or renders its text differently)
    unknown            the page was not read

A page that changed is not an error in the set: sources are live pages, and each receipt records when it
was read. The exit code is 0 unless --strict is given and a receipt no longer matches or was not fetched.
One request per URL, one at a time, with a pause between them; fetched content is only searched, never run.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from near_miss import tidy  # noqa: E402  (the step that made the shown passages)

AGENT = "evidence-verdicts-receipt-check/1 (open benchmark tooling; one request per source)"


def to_text(html):
    """The HTML-to-text step the receipts were saved with."""
    s = re.sub(r"<(script|style|noscript|svg)\b[\s\S]*?</\1>", " ", html, flags=re.I)
    s = re.sub(r"</(p|div|li|tr|h[1-6]|section|article|br)>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        s = s.replace(entity, char)
    s = re.sub(r"&#39;|&#x27;|&rsquo;", "'", s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def matches(pattern, text):
    """Receipts are tested without regard to case, on text whose whitespace runs are single spaces."""
    return re.search(pattern, re.sub(r"\s+", " ", text), flags=re.I) is not None


def fetch(url, timeout=40):
    if not url.startswith("https://"):
        raise ValueError("not an https URL")
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get("content-type") or ""


def receipts_of(folder):
    """Every passage a prompt of the set shows, once: {(url, pattern): row}, with the passage's text."""
    folder = Path(folder)
    shown = {}
    for r in json.loads((folder / "records.json").read_text(encoding="utf-8"))["records"]:
        for x in r["receipts"] + ([r["near"]] if r.get("near") else []):
            if x.get("excerpt"):
                shown[(x["key"], x["pattern"])] = x["excerpt"]
    out = {}
    for item in json.loads((folder / "provenance.json").read_text(encoding="utf-8")).values():
        for e in item["evidence"]:
            row = {k: e[k] for k in ("key", "url", "host", "sha256", "fetched_at", "pattern")}
            out.setdefault((e["url"], e["pattern"]), {**row, "passage": shown[(e["key"], e["pattern"])]})
    return out


def check(rows, pause=0.5, get=fetch, say=lambda line: print(line, file=sys.stderr)):
    pages = {}
    for row in rows:
        url = row["url"]
        if url not in pages:
            if pages:
                time.sleep(pause)
            try:
                raw, kind = get(url)
                decoded = raw.decode("utf-8", errors="replace")
                text = to_text(decoded) if "html" in kind else decoded
                pages[url] = {"sha256": hashlib.sha256(raw).hexdigest(), "text": text, "shown": tidy(re.sub(r"\s+", " ", text)),
                              "pdf": raw[:5] == b"%PDF-" or "pdf" in kind}
            except (urllib.error.URLError, OSError, ValueError) as error:
                pages[url] = {"error": str(getattr(error, "code", "") or error)[:80]}
            say(f"{len(pages):>4} {pages[url].get('error') or pages[url]['sha256'][:12]} {url}")
        page = pages[url]
        if "error" in page:
            row["today"], row["error"], row["passage_today"] = "not fetched", page["error"], "unknown"
            continue
        if page["sha256"] == row["sha256"]:
            row["today"] = "same bytes"
        elif page["pdf"]:
            row["today"] = "a PDF, not read"
        else:
            row["today"] = "still matches" if matches(row["pattern"], page["text"]) else "no longer matches"
        row["passage_today"] = "unknown" if page["pdf"] else "on the page" if row["passage"] in page["shown"] else "not on the page"
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Test every passage the set shows on its source as it is today.")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent), help="folder with records.json and provenance.json")
    ap.add_argument("--limit", type=int, help="only the first N receipts")
    ap.add_argument("--out", help="write the per-receipt result here")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a receipt no longer matches or was not fetched")
    a = ap.parse_args(argv)
    rows = list(receipts_of(a.dir).values())[:a.limit]
    check(rows)
    tally = lambda field: dict(sorted({v: sum(r[field] == v for r in rows) for v in {r[field] for r in rows}}.items()))
    summary = {"receipts": len(rows), "sources": len({r["url"] for r in rows}), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "today": tally("today"), "passage_today": tally("passage_today")}
    if a.out:
        slim = [{k: v for k, v in r.items() if k != "passage"} for r in rows]
        Path(a.out).write_text(json.dumps({**summary, "rows": slim}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    bad = summary["today"].get("no longer matches", 0) + summary["today"].get("not fetched", 0)
    return 1 if a.strict and bad else 0


if __name__ == "__main__":
    sys.exit(main())
