#!/usr/bin/env python3
"""Rebuilds "Evidence verdicts" from its records, byte for byte. Standard library only.

    python build_set.py DIR            rebuild from DIR's records.json, config.json, solvers.json and
                                       handcheck.json, and compare with DIR's committed set and hash
    python build_set.py DIR --out OUT  also write the rebuilt files into OUT
    python build_set.py DIR --candidates FILE
                                       also write every candidate, before the gate, for ask_models.py

Every choice is made by a sha256 order key, numbers are compared as exact decimals, and every text rule
is spelled out over ASCII, so a rebuild gives the same bytes on any machine and Python version.

Three families, one label each, so no single answer scores:
  ev_supported     a true claim with passages that state it
  ev_refuted       a stale or wrong claim with passages that show today's value, or a true claim with
                   one number (for past events, one year) changed by the rule in numeric_edits/year_edits
  ev_insufficient  a true claim with passages that do not settle it: passages saved for another claim
                   (insufficient_pairs), or one close passage (two or more content words in common, none of
                   the claim's values): a sentence of the claim's own source (the record's `near`, chosen by
                   near_miss.py and checked again here) or a passage of another record of the same article
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

NS = "evidence-verdicts-v1"
FAMILIES = ["ev_supported", "ev_refuted", "ev_insufficient"]
LABEL = {"ev_supported": "supported", "ev_refuted": "refuted", "ev_insufficient": "insufficient"}
A = re.ASCII


# ---------- hashes and order ----------

def sha256hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def order_key(*parts):
    return sha256hex("|".join([NS, *parts]))


def by_key(items, key_of):
    """A copy sorted by an order key; equal keys keep their order."""
    return [x for _, _, x in sorted(((key_of(x), i, x) for i, x in enumerate(items)), key=lambda t: (t[0], t[1]))]


# ---------- numbers ----------

SCALES = [("thousand", 3), ("million", 6), ("billion", 9), ("trillion", 12)]
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
YEAR_WORDS = {"year", "in", "by", "since", "until", "before", "after", "circa", "around", "during"}
SPACES = {" ", "\u00a0", "\u202f", "\u2009"}


def is_digit(c):
    return "0" <= c <= "9"


def is_letter(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def at(text, i):
    return text[i] if 0 <= i < len(text) else ""


def canonical(int_part, frac, scale):
    """The exact decimal string for digits written as int_part and frac, times ten to the scale."""
    digits = int_part + frac
    exp = scale - len(frac)
    if exp >= 0:
        whole, part = digits + "0" * exp, ""
    else:
        cut = len(digits) + exp
        whole = digits[:cut] if cut > 0 else ""
        part = ("0" * -cut if cut < 0 else "") + digits[max(0, cut):]
    whole = whole.lstrip("0")
    part = part.rstrip("0")
    return (whole or "0") + ("." + part if part else "")


def common(a, b):
    """Two canonical decimals as integers on one scale, for exact ratio tests."""
    fa = len(a) - a.index(".") - 1 if "." in a else 0
    fb = len(b) - b.index(".") - 1 if "." in b else 0
    f = max(fa, fb)
    return int(a.replace(".", "") + "0" * (f - fa)), int(b.replace(".", "") + "0" * (f - fb))


def word_before(text, i):
    e = i
    while e > 0 and (text[e - 1] in SPACES or text[e - 1] == ","):
        e -= 1
    s = e
    while s > 0 and is_letter(text[s - 1]):
        s -= 1
    return text[s:e].lower()


def word_after(text, i):
    s = i
    while s < len(text) and (text[s] in SPACES or text[s] == ","):
        s += 1
    e = s
    while e < len(text) and is_letter(text[e]):
        e += 1
    return text[s:e].lower()


def number_tokens(text):
    """Every number in the text, in order, with its kind: number, year, day, one, ordinal or id."""
    out = []
    n_text = len(text)
    i = 0
    while i < n_text:
        if not is_digit(text[i]):
            i += 1
            continue
        num_start = i
        j = i
        while j < n_text and is_digit(text[j]):
            j += 1
        grouped = False
        if j - i <= 3:
            k = j
            while (k + 3 < n_text and text[k] == "," and is_digit(text[k + 1]) and is_digit(text[k + 2]) and is_digit(text[k + 3])
                   and not (k + 4 < n_text and is_digit(text[k + 4]))):
                k += 4
                grouped = True
            if grouped:
                j = k
        int_part = text[i:j].replace(",", "")
        frac = ""
        if j + 1 < n_text and text[j] == "." and is_digit(text[j + 1]):
            k = j + 1
            while k < n_text and is_digit(text[k]):
                k += 1
            frac = text[j + 1:k]
            j = k
        num_end = j
        before, before2 = at(text, num_start - 1), at(text, num_start - 2)
        after, after2 = at(text, num_end), at(text, num_end + 1)
        kind = "number"
        prev = out[-1] if out else None
        if (before and (is_letter(before) or is_digit(before))) or before in ("#", "/", "_"):
            kind = "id"
        elif before == "-" and before2 and is_letter(before2):
            kind = "id"
        elif (before == "-" and prev is not None and prev["kind"] == "id" and prev["numEnd"] == num_start - 1
              and is_letter(at(text, prev["numStart"] - 1) or " ")):
            kind = "id"  # a name that starts with letters, continued after a hyphen: x86-64
        elif before == "." and before2 and is_digit(before2):
            kind = "id"
        elif after == "/" and after2 and is_digit(after2):
            kind = "id"
        elif after == "." and after2 and is_digit(after2):
            kind = "id"
        elif after == "_" or (after == "-" and after2 and (is_letter(after2) or is_digit(after2))):
            kind = "id"
        elif after and is_letter(after):
            kind = "ordinal" if word_after(text, num_end) in ("st", "nd", "rd", "th") else "id"
        end, scale, unit = num_end, 0, ""
        if kind == "number":
            k = num_end
            if k < n_text and text[k] in SPACES:
                k += 1
            rest = text[k:k + 9].lower()
            for word, exp in SCALES:
                nxt = at(text, k + len(word))
                if rest.startswith(word) and not (nxt and is_letter(nxt)):
                    scale, end = exp, k + len(word)
                    break
            u = end
            if u < n_text and text[u] in SPACES:
                u += 1
            if at(text, u) == "%":
                unit, end = "%", u + 1
            elif word_after(text, end) == "percent":
                unit, end = "%", text.lower().find("percent", end) + 7
        n = int(int_part) if (frac == "" and not grouped) else None
        if kind == "number" and scale == 0 and unit == "":
            plain4 = n is not None and len(int_part) == 4
            if plain4 and 1900 <= n <= 2099:
                kind = "year"
            elif plain4 and 1000 <= n <= 2999 and word_before(text, num_start) in YEAR_WORDS:
                kind = "year"
            elif n is not None and 1 <= n <= 31 and (word_before(text, num_start) in MONTHS or word_after(text, num_end) in MONTHS):
                kind = "day"
            elif frac == "" and int_part == "1":
                kind = "one"
        out.append({
            "start": num_start, "end": end, "numStart": num_start, "numEnd": num_end, "raw": text[num_start:end], "written": text[num_start:num_end],
            "int": int_part, "frac": frac, "scale": scale, "unit": unit,
            "value": canonical(int_part, frac, scale) if kind in ("number", "year") else None, "kind": kind,
        })
        i = max(end, num_end)
    return out


def value_set(text):
    return {f"{t['value']}|{t['unit']}" for t in number_tokens(text) if t["value"] is not None}


def value_kinds(text):
    """The kinds of the quantities and years in the text: year, percent, scaled (with a scale word) or plain."""
    return {"year" if t["kind"] == "year" else "percent" if t["unit"] == "%" else "scaled" if t["scale"] > 0 else "plain"
            for t in number_tokens(text) if t["value"] is not None}


def bare_digits(text):
    """The bare digits of every number written in `text` that has two or more of them ("81.7M" gives "817")."""
    out = []
    for m in re.finditer(r"[0-9][0-9.,]*[0-9]|[0-9]", str(text)):
        d = re.sub(r"[^0-9]+", "", m.group(0))
        if len(d) >= 2 and d not in out:
            out.append(d)
    return out


# ---------- the refuting edit ----------

BOUND_BEFORE = [
    "more than", "over", "at least", "up to", "under", "fewer than", "less than", "above", "below",
    "exceeding", "exceeds", "exceeded", "exceed", "at most", "as many as", "as much as", "greater than", "upwards of",
    "minimum of", "maximum of", "no more than", "no less than", "no fewer than", "within",
]
BOUND_AFTER = ["or more", "or less", "or fewer", "or greater", "or higher", "or lower", "and above", "and over", "plus"]
YEAR_CUE_BEFORE = ["since", "by", "until", "till", "before", "after", "from", "between", "through", "to", "of", "around", "circa"]
CURRENCY = ["$", "\u20ac", "\u00a3", "\u20bf", "\u00a5", "\u20ae"]
CUR = "[$\u20ac\u00a3\u20bf\u00a5]?"
DASH = "\\-\u2013\u2014"


def lower_before(text, i):
    e = i
    while e > 0 and (text[e - 1] in CURRENCY or text[e - 1] in (" ", "\u00a0", "~", "(")):
        e -= 1
    return text[max(0, e - 40):e].lower()


def lower_after(text, i):
    return re.sub(r"^[\s\u00a0]+", "", text[i:i + 24].lower(), flags=A)


def ends_with_word(s, cue):
    return s.endswith(cue) and (len(s) == len(cue) or not ("a" <= s[len(s) - len(cue) - 1] <= "z"))


def is_day(w):
    return re.fullmatch(r"[0-9]{1,2}(?:st|nd|rd|th)?", w, A) is not None


def without_date(s):
    """The words before a year with its date taken off, so that a cue word still counts when a month and a day
    stand between it and the year: "until June 2020", "as of December 31, 2025", "through 3 March 2024"."""
    s = s.rstrip()
    words = (s[:-1] if s.endswith(",") else s).split(" ")
    last = lambda: re.sub(r"[.,]$", "", words[-1] if words else "")
    if is_day(last()):
        words.pop()
    if last() in MONTHS or re.fullmatch(r"q[1-4]|h[12]", last(), A):  # a month, or a quarter or half: "through Q1 2024"
        words.pop()
        if is_day(last()):
            words.pop()
    out = " ".join(words).rstrip()
    return out[:-1] if out.endswith(",") else out


def starts_with_word(s, cue):
    return s.startswith(cue) and (len(s) == len(cue) or not ("a" <= s[len(cue)] <= "z"))


def in_range(text, t):
    after = text[t["end"]:t["end"] + 24]
    before = text[max(0, t["numStart"] - 40):t["numStart"]]
    # one word may stand between an end and the dash or "to": "3 million users to 9 million users"
    if re.search(rf"^(?:\s[a-z]+)?\s?(?:[{DASH}]|to)\s?{CUR}\d", after, A):
        return True
    # the near end may carry a scale word or "percent", and that word: "5 million to 10 million", "5 percent to 10 percent"
    if re.search(rf"\d(?:[a-z%]*|\s(?:thousand|million|billion|trillion|percent))(?:\s[a-z]+)?\s?(?:[{DASH}]|to)\s?{CUR}$", before, A | re.I):
        return True
    lead = text[max(0, t["numStart"] - 60):t["numStart"]].lower()
    if re.search(r"\bbetween\b[^.;]*$", lead, A) and re.search(rf"^\s?and\s?{CUR}\d", after, A):
        return True
    if re.search(rf"\bbetween\b[^.;]*\d[a-z%,. ]*\sand\s{CUR}$", lead, A):
        return True
    return False


def replace_leading_digit(written, digit):
    for i, ch in enumerate(written):
        if "1" <= ch <= "9":
            return ch, written[:i] + digit + written[i + 1:]
    return None


def numeric_edits(rec_id, claim, excerpts, max_edits=2):
    toks = number_tokens(claim)
    idxs = [i for i, t in enumerate(toks) if t["kind"] == "number"]
    skipped = []
    if not idxs:
        return [], skipped
    if len(idxs) >= 3:
        return [], [{"token": i, "raw": toks[i]["raw"], "why": "the claim lists three or more quantities"} for i in idxs]
    in_excerpts = set()
    for e in excerpts:
        in_excerpts |= value_set(e)
    digits_of_excerpts = [re.sub(r"[^0-9]", "", e) for e in excerpts]
    count = {}
    for t in toks:
        if t["value"] is not None:
            k = f"{t['value']}|{t['unit']}"
            count[k] = count.get(k, 0) + 1
    edits = []
    for i in by_key(idxs, lambda k: order_key("cf-token", rec_id, str(k))):
        t = toks[i]
        key = f"{t['value']}|{t['unit']}"

        def no(why, i=i, t=t):
            skipped.append({"token": i, "raw": t["raw"], "why": why})
        if count.get(key, 0) > 1:
            no("the value is said twice in the claim")
            continue
        if in_range(claim, t):
            no("the value is one end of a range")
            continue
        before = lower_before(claim, t["numStart"]).rstrip()
        if any(ends_with_word(before, c) for c in BOUND_BEFORE):
            no("the value follows a one-sided bound")
            continue
        after = lower_after(claim, t["end"])
        if any(starts_with_word(after, c) for c in BOUND_AFTER) or at(claim, t["end"]) == "+":
            no("the value is followed by a bound")
            continue
        if key not in in_excerpts:
            no("the value is not in the excerpts")
            continue
        lead = replace_leading_digit(t["written"], "0")
        if not lead:
            no("the value is zero")
            continue
        done = None
        for d in by_key([x for x in "123456789" if x != lead[0]], lambda x: order_key("cf", rec_id, str(i), x)):
            written = replace_leading_digit(t["written"], d)[1]
            int_part, _, frac = written.partition(".")
            int_part = int_part.replace(",", "")
            to_value = canonical(int_part, frac, t["scale"])
            a, b = common(to_value, t["value"])
            if not (4 * a <= 3 * b or 4 * a >= 5 * b):
                continue
            if t["unit"] == "%":
                x, hundred = common(to_value, "100")
                if x > hundred:
                    continue
            nk = f"{to_value}|{t['unit']}"
            if nk in in_excerpts or nk in count:
                continue
            bare = int_part + frac
            if len(bare) >= 3 and any(bare in x for x in digits_of_excerpts):
                continue
            to = written + claim[t["numEnd"]:t["end"]]
            done = {"token": i, "from": t["raw"], "to": to, "fromValue": t["value"], "toValue": to_value, "unit": t["unit"],
                    "claim": claim[:t["numStart"]] + to + claim[t["end"]:]}
            break
        if not done:
            no("no replacement digit passes the safeguards")
            continue
        edits.append(done)
        if len(edits) >= max_edits:
            break
    return edits, skipped


def year_edits(rec_id, claim, excerpts, max_edits=1, min_year=1990, max_year=2026):
    toks = number_tokens(claim)
    idxs = [i for i, t in enumerate(toks) if t["kind"] == "year"]
    skipped = []
    in_excerpts = set()
    for e in excerpts:
        in_excerpts |= value_set(e)
    count = {}
    for t in toks:
        if t["value"] is not None:
            k = f"{t['value']}|{t['unit']}"
            count[k] = count.get(k, 0) + 1
    edits = []
    for i in by_key(idxs, lambda k: order_key("cf-token", rec_id, str(k))):
        t = toks[i]
        key = f"{t['value']}|"

        def no(why, i=i, t=t):
            skipped.append({"token": i, "raw": t["raw"], "why": why})
        if count.get(key, 0) > 1:
            no("the year is said twice in the claim")
            continue
        if in_range(claim, t):
            no("the year is one end of a range")
            continue
        before = without_date(lower_before(claim, t["numStart"]).rstrip())
        if any(ends_with_word(before, c) for c in YEAR_CUE_BEFORE):
            no("the year follows a word that makes it a bound or a span")
            continue
        if key not in in_excerpts:
            no("the year is not in the excerpts")
            continue
        y = int(t["value"])
        cands = [str(y + s * k) for k in (2, 3, 4, 5, 6, 7) for s in (-1, 1)]
        done = None
        for c in by_key(cands, lambda x: order_key("cf-year", rec_id, str(i), x)):
            n = int(c)
            if n < min_year or n > max_year:
                continue
            if f"{c}|" in in_excerpts or f"{c}|" in count:
                continue
            done = {"token": i, "from": t["raw"], "to": c, "fromValue": t["value"], "toValue": c, "unit": "",
                    "claim": claim[:t["numStart"]] + c + claim[t["end"]:]}
            break
        if not done:
            no("no replacement year passes the safeguards")
            continue
        edits.append(done)
        if len(edits) >= max_edits:
            break
    return edits, skipped


# ---------- text ----------

# what JavaScript's \s matches, so that both builders collapse the same characters
JS_SPACE = "[\\t\\n\\v\\f\\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+"


def clean(s):
    return re.sub(JS_SPACE, " ", "" if s is None else str(s)).strip(" ")


STOP = set((
    "a an the and or but if then else so than that this these those there here it its is are was were be been being has have had do does did will would can could may might must shall should "
    "of in on at by for from to with without into onto over under about above below after before between through during per via as not no nor only also both each every any some such more most less least many much few "
    "which who whom whose what when where why how one two three first second new other same own very just still even now then also their his her our your they them he she we you i "
    "used use uses using called known based include includes including included named make makes made allow allows allowed provide provides provided").split(" "))


def content_words(text, title_words=frozenset()):
    """Lowercase ASCII letter runs of four or more letters, minus stop words and title words, a final s dropped."""
    out = []
    seen = set()
    w = ""

    def push():
        nonlocal w
        if len(w) >= 4 and w not in STOP:
            stem = w[:-1] if (len(w) > 4 and w.endswith("s") and not w.endswith("ss")) else w
            if stem not in title_words and w not in title_words and stem not in seen:
                seen.add(stem)
                out.append(stem)
        w = ""
    for c in str(text).lower():
        if "a" <= c <= "z":
            w += c
        else:
            push()
    push()
    return out


def title_words_of(title):
    return set(content_words(re.sub(r"\(.*?\)", " ", str(title))))


OPENERS = ["however", "therefore", "thus", "hence", "instead", "furthermore", "moreover", "additionally", "similarly", "consequently", "nevertheless",
           "nonetheless", "otherwise", "meanwhile", "likewise", "as a result", "for example", "for instance", "in addition", "in contrast",
           "on the other hand", "he", "she", "they", "these", "those", "both", "the latter", "the former"]


def wording_veto(text, people_re=None, min_words=5):
    t = clean(text)
    if len(t.split(" ")) < min_words:
        return "too few words"
    low = t.lower()
    for o in OPENERS:
        nxt = low[len(o)] if len(low) > len(o) else ""
        if low.startswith(o) and not ("a" <= nxt <= "z"):
            return "opens with a word that points outside the sentence"
    if people_re is not None and people_re.search(t):
        return "names a person"
    return None


def grams(text, n=10):
    w = [x for x in re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split(" ") if x]
    out = []
    seen = set()
    for i in range(0, len(w) - n + 1):
        g = " ".join(w[i:i + n])
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


# ---------- "not enough evidence" pairs ----------

def insufficient_pairs(pool, cap=None):
    words, nums, ex_words, ex_nums = {}, {}, {}, {}
    for r in pool:
        tw = title_words_of(r["title"])
        words[r["id"]] = content_words(r["claim"], tw)
        nums[r["id"]] = value_set(r["claim"])
        ew, en = set(), set()
        for e in r["excerpts"]:
            ew |= set(content_words(e["text"], tw))
            en |= value_set(e["text"])
        ex_words[r["id"]], ex_nums[r["id"]] = ew, en
    used = set()
    pairs, unpaired = [], []
    n = 0
    for c in by_key(pool, lambda r: order_key("ins-claim", r["id"])):
        if cap is not None and len(pairs) >= cap:
            break
        want = "same_page" if n % 2 == 0 else "cross_page"

        def allowed(d, stratum, c=c):
            if d["id"] == c["id"] or d["id"] in used:
                return None
            if (d["slug"] == c["slug"]) != (stratum == "same_page"):
                return None
            if any(u in c["urls"] for u in d["urls"]):
                return None
            if nums[c["id"]] & ex_nums[d["id"]]:
                return None
            tw = title_words_of(c["title"]) | title_words_of(d["title"])
            shared = sorted(w for w in words[c["id"]] if w not in tw and w in ex_words[d["id"]])
            if len(shared) > (1 if stratum == "same_page" else 0):
                return None
            return shared
        found = None
        for stratum in (want, "cross_page" if want == "same_page" else "same_page"):
            for d in by_key(pool, lambda r: order_key("ins-donor", c["id"], r["id"])):
                shared = allowed(d, stratum)
                if shared is not None:
                    found = {"claimId": c["id"], "donorId": d["id"], "stratum": stratum, "shared": shared}
                    break
            if found:
                break
        if not found:
            unpaired.append({"claimId": c["id"], "why": "no record may lend it excerpts under the pairing rules"})
            continue
        used.add(found["donorId"])
        pairs.append(found)
        n += 1
    return pairs, unpaired


# ---------- prompts ----------

ORDERS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
LETTERS = ["A", "B", "C"]


def order_of(item_key):
    return ORDERS[int(order_key("options", item_key)[:2], 16) % 6]


def render(cfg, title, claim, excerpts, order, label):
    labels = [cfg["labels"][i] for i in order]
    n = len(excerpts)
    lines = [
        f'Claim, about the subject of the English Wikipedia article "{title}":',
        claim,
        "",
        f'Evidence ({n} {"excerpt from another website" if n == 1 else "excerpts from other websites"}):',
        *[f'[{i + 1}] {e["host"]}: "{e["text"]}"' for i, e in enumerate(excerpts)],
        "",
        "Going only by the evidence above, and not by what you already know, the claim is:",
        *[f"{LETTERS[i]}. {cfg['options'][lab]}" for i, lab in enumerate(labels)],
        cfg["instruction"],
    ]
    return "\n".join(lines), LETTERS[labels.index(label)], labels


def read_choice(output):
    """Delta League's `choice` scorer: the last \\boxed{X}, else a bare letter alone on the last line."""
    text = ("" if output is None else str(output)).split("</think>")[-1].strip()  # thinking traces do not count
    boxed = re.findall(r"\\boxed\{([A-E1-5])\}", text)
    matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", text)
    return matches[-1] if matches else None


# ---------- the builder ----------

def claim_text(r, cfg):
    t = clean(r["text"] if cfg.get("claim_form") == "sentence" else r["claim"])
    return t if re.search(r"[.!?]$", t) else t + "."


def candidates(records, cfg):
    people = re.compile(cfg["people"], A)
    vetoed = []

    def veto(key, family, cluster, why, **extra):
        vetoed.append({"key": key, "family": family, "cluster": cluster, "why": why, **extra})

    def excluded(x):
        host, url = x.get("host") or "", str(x.get("url"))
        return any(host == h or host.endswith("." + h) or f"/{h}/" in url or f"//{h}" in url or f".{h}/" in url for h in cfg["excluded_hosts"])

    ready = []
    for r in records:
        shown, dropped = [], []
        for x in r["receipts"]:
            if x["kind"] not in cfg["kinds_in_prompts"]:
                dropped.append({"key": x["key"], "why": f"a {x['kind']} receipt, not shown in prompts"})
                continue
            if excluded(x):
                dropped.append({"key": x["key"], "why": "a news publisher's page"})
                continue
            if not x.get("sha256") or x.get("status") != 200:
                dropped.append({"key": x["key"], "why": "no saved hash for this fetch"})
                continue
            text = clean(x.get("excerpt"))
            if not text or len(text) > cfg["caps"]["excerpt_chars"] or len(text.split(" ")) < 4:
                dropped.append({"key": x["key"], "why": "no excerpt of a usable length"})
                continue
            if any(s["text"] == text for s in shown):
                dropped.append({"key": x["key"], "why": "the same excerpt as an earlier receipt"})
                continue
            shown.append({"key": x["key"], "url": x["url"], "host": x["host"], "kind": x["kind"], "fetched_at": x["fetched_at"],
                          "sha256": x["sha256"], "pattern": x["pattern"], "cut": x["cut"], "text": text})
        claim = claim_text(r, cfg)
        why = wording_veto(claim, people, cfg["caps"]["claim_min_words"])
        if not why and len(claim) > cfg["caps"]["claim_chars"]:
            why = "the claim is longer than the cap"
        if not why and not shown:
            why = f"no receipt can be shown ({'; '.join(d['why'] for d in dropped) or 'none saved'})"
        if why:
            veto(f"rec|{r['id']}", "ev_supported" if r["verdict"] == "current" else "ev_refuted", r["id"], why)
            continue
        ready.append({**r, "claim": claim, "shown": shown[:cfg["caps"]["excerpts_per_item"]]})

    items = []
    form = "sentence" if cfg.get("claim_form") == "sentence" else "checkable"

    def make(**it):
        order = order_of(it["key"])
        prompt, answer, options = render(cfg, it["title"], it["claim"], it["excerpts"], order, LABEL[it["family"]])
        return {**it, "form": form, "label": LABEL[it["family"]], "order": order, "options": options, "prompt": prompt, "answer": answer, "sha": sha256hex(prompt)}

    def add(**it):
        items.append(make(**it))

    trues = [r for r in ready if r["verdict"] == "current"]
    for r in trues:
        add(key=f"sup|{r['id']}", family="ev_supported", stratum="extra_check" if r["source"] == "extra" else "corpus", cluster=r["id"],
            title=r["article"]["title"], claim=r["claim"], excerpts=r["shown"], record=r["id"], donor=None, edit=None)
    for r in ready:
        if r["verdict"] != "current":
            add(key=f"ref|{r['id']}|natural", family="ev_refuted", stratum=r["verdict"], cluster=r["id"],
                title=r["article"]["title"], claim=r["claim"], excerpts=r["shown"], record=r["id"], donor=None, edit=None)
    for r in trues:
        texts = [s["text"] for s in r["shown"]]
        num, num_skipped = numeric_edits(r["id"], r["claim"], texts, cfg["caps"]["edits_per_sentence"])
        if r["class"] == "sourced":
            yr, yr_skipped = year_edits(r["id"], r["claim"], texts, 1, cfg["years"]["min"], cfg["years"]["max"])
        else:
            yr, yr_skipped = [], []
        every = ([{**e, "rule": "leading-digit"} for e in num] + [{**e, "rule": "year-shift"} for e in yr])[:cfg["caps"]["edits_per_sentence"]]
        for e in every:
            add(key=f"ref|{r['id']}|{e['rule']}|{e['token']}", family="ev_refuted", stratum="year_edit" if e["rule"] == "year-shift" else "numeric_edit",
                cluster=r["id"], title=r["article"]["title"], claim=e["claim"], excerpts=r["shown"], record=r["id"], donor=None, original=r["claim"],
                edit={"rule": e["rule"], "token": e["token"], "from": e["from"], "to": e["to"], "from_value": e["fromValue"], "to_value": e["toValue"], "unit": e["unit"]},
                needs=f"sup|{r['id']}")
        for s in num_skipped + yr_skipped:
            veto(f"ref|{r['id']}|token{s['token']}", "ev_refuted", r["id"], s["why"], raw=s["raw"])

    pool = [{"id": r["id"], "slug": r["article"]["slug"], "title": r["article"]["title"], "claim": r["claim"],
             "urls": [x["url"] for x in r["receipts"]], "excerpts": r["shown"]} for r in trues]
    pairs, unpaired = insufficient_pairs(pool)
    by_id = {r["id"]: r for r in trues}
    for p in pairs:
        c, d = by_id[p["claimId"]], by_id[p["donorId"]]
        add(key=f"ins|{c['id']}|{d['id']}", family="ev_insufficient", stratum=p["stratum"], cluster=c["id"], title=c["article"]["title"],
            claim=c["claim"], excerpts=d["shown"], record=c["id"], donor=d["id"], edit=None, shared_words=p["shared"])
    for u in unpaired:
        veto(f"ins|{u['claimId']}", "ev_insufficient", u["claimId"], u["why"])

    kept = []
    for fam in FAMILIES:
        seen_body, seen_grams = {}, {}
        for it in by_key([x for x in items if x["family"] == fam], lambda x: order_key("item", x["key"])):
            body = (it["claim"] + " " + " ".join(e["text"] for e in it["excerpts"])).lower()
            body = re.sub(r" +", " ", re.sub(JS_SPACE, " ", re.sub(r"[^a-z0-9 ]+", " ", body))).strip(" ")[:90]
            if body in seen_body:
                veto(it["key"], fam, it["cluster"], f"duplicate body of {seen_body[body]}")
                continue
            g = grams(it["claim"])
            clash = None
            for x in g:
                k = seen_grams.get(x)
                if k and k["cluster"] != it["cluster"]:
                    clash = k["key"]
                    break
            if clash:
                veto(it["key"], fam, it["cluster"], f"its claim shares a 10-word sequence with {clash}")
                continue
            seen_body[body] = it["key"]
            for x in g:
                seen_grams.setdefault(x, {"key": it["key"], "cluster": it["cluster"]})
            kept.append(it)

    # The harder part of the third family: a passage that talks about the claim's subject and does not
    # state the claimed value. Only a claim that holds a quantity or a year gets one, so that "holds none of
    # the claim's values" can be tested: the passage shares at least two content words with the claim, and
    # holds none of its values nor the bare digits of any of its numbers. It is another sentence of the
    # claim's own source (near_miss, found in the saved page when the records were made) or a passage of
    # another record of the same article, from another page (related; a record lends once). These items
    # are added after the passes above, so they never displace an item of the easier strata.
    hard = []
    valued = [r for r in trues if value_set(r["claim"])]

    ignore = set(cfg.get("close_ignore_words") or [])  # scale and approximation words say how big a number is, not what it is about

    def close_to(r, text):
        """The close-passage rule, checked on the text a prompt shows: shared content words, whether the passage
        holds one of the claim's values or the bare digits of one, and whether it holds a decoy (another value
        of the same kind as one of the claim's)."""
        tw = title_words_of(r["article"]["title"])
        words, kinds = set(content_words(r["claim"], tw)), value_kinds(r["claim"])
        held, bare = value_set(text), re.sub(r"[^0-9]+", "", text)
        return {"shared": sorted(w for w in content_words(text, tw) if w in words and w not in ignore),
                "holds_value": bool(value_set(r["claim"]) & held) or any(d in bare for d in bare_digits(r["claim"])),
                "decoy": bool(value_kinds(text) & kinds)}

    def better(c, best):
        return best is None or (c["decoy"] and not best["decoy"]) or (c["decoy"] == best["decoy"] and len(c["shared"]) > len(best["shared"]))

    for r in valued:
        n = r.get("near")
        if not n:
            continue
        text = clean(n.get("excerpt"))
        c = close_to(r, text) if text else None
        why = None
        if n["kind"] not in cfg["kinds_in_prompts"] or excluded(n) or not n.get("sha256") or n.get("status") != 200:
            why = "its near-miss passage is from a source that prompts do not show"
        elif not text or len(text) > cfg["caps"]["excerpt_chars"] or len(text.split(" ")) < 4:
            why = "its near-miss passage is not of a usable length"
        elif people.search(text):
            why = "its near-miss passage names a person"
        elif len(c["shared"]) < 2:
            why = "its near-miss passage shares fewer than two content words with the claim"
        elif c["holds_value"]:
            why = "its near-miss passage holds one of the claim's values"
        elif any(text in s["text"] or s["text"] in text for s in r["shown"]):
            why = "its near-miss passage repeats a passage the claim's own item shows"
        if why:
            veto(f"ins|{r['id']}|near", "ev_insufficient", r["id"], why)
            continue
        passage = {"key": n["key"], "url": n["url"], "host": n["host"], "kind": n["kind"], "fetched_at": n["fetched_at"],
                   "sha256": n["sha256"], "pattern": n["pattern"], "cut": "near", "text": text}
        hard.append(make(key=f"ins|{r['id']}|near", family="ev_insufficient", stratum="near_miss", cluster=r["id"], title=r["article"]["title"],
                         claim=r["claim"], excerpts=[passage], record=r["id"], donor=r["id"], edit=None, shared_words=c["shared"], decoy=c["decoy"]))
    lent = set()
    for r in by_key(valued, lambda x: order_key("rel-claim", x["id"])):
        urls = {x["url"] for x in r["receipts"]}
        best = None
        donors = [x for x in trues if x["id"] != r["id"] and x["article"]["slug"] == r["article"]["slug"] and x["id"] not in lent]
        for d in by_key(donors, lambda x: order_key("rel-donor", r["id"], x["id"])):
            for passage in d["shown"]:
                if passage["url"] in urls or people.search(passage["text"]):
                    continue
                c = close_to(r, passage["text"])
                if len(c["shared"]) >= 2 and not c["holds_value"] and better(c, best):
                    best = {**c, "d": d, "s": passage}
        if best is None:
            continue
        lent.add(best["d"]["id"])
        hard.append(make(key=f"ins|{r['id']}|rel|{best['d']['id']}", family="ev_insufficient", stratum="related", cluster=r["id"],
                         title=r["article"]["title"], claim=r["claim"], excerpts=[best["s"]], record=r["id"], donor=best["d"]["id"], edit=None,
                         shared_words=best["shared"], decoy=best["decoy"]))
    kept.extend(by_key(hard, lambda x: order_key("item", x["key"])))
    return kept, vetoed, len(ready)


HARD = {"near_miss", "related"}  # the strata of the third family whose passage is close to the claim: two or more shared content words


def shape(items, id_of):
    return {fam: [{"id": id_of(x), "prompt": x["prompt"], "answer": x["answer"], "scorer": "choice", "samples": 1} for x in items if x["family"] == fam] for fam in FAMILIES}


def provenance_of(it, records_by_id):
    r = records_by_id[it["record"]]
    a = r["article"]
    out = {
        "key": it["key"], "family": it["family"][3:], "stratum": it["stratum"], "label": it["label"], "options": it["options"], "cluster": it["cluster"],
        "prompt_sha256": it["sha"],
        "claim": {"form": it["form"], "modified": True, "original": it["original"], "edit": it["edit"]} if it["edit"] else {"form": it["form"], "modified": False},
        "wikipedia": {"title": a["title"], "revid": a["revid"], "timestamp": a["timestamp"], "url": a["url"], "licence": a["licence"], "sentence_id": it["record"]},
        "evidence_from": it["donor"] or it["record"],
        "evidence": [{"key": e["key"], "url": e["url"], "host": e["host"], "kind": e["kind"], "fetched_at": e["fetched_at"], "sha256": e["sha256"],
                      "pattern": e["pattern"], "cut": e["cut"]} for e in it["excerpts"]],
    }
    if it.get("shared_words") is not None:
        out["shared_words"] = it["shared_words"]
    if it.get("decoy") is not None:
        out["decoy"] = it["decoy"]
    return out


def gate_passed(items, cfg, solvers):
    gate = cfg["gate_models"]
    dropped = []
    models = (solvers or {}).get("models", {})

    def letter_of(model, it):
        a = models.get(model, {}).get(it["sha"])
        return a.get("letter") if a else None
    by_key_of = {x["key"]: x for x in items}
    ok = {}
    for it in items:
        answers = {m: letter_of(m, it) for m in gate}
        if any(it["sha"] not in models.get(m, {}) for m in gate):
            dropped.append({"key": it["key"], "family": it["family"], "why": "a gate model has not been asked this prompt", "answers": answers})
            continue
        if any(answers[m] is None for m in gate):
            dropped.append({"key": it["key"], "family": it["family"], "why": "a gate model gave no letter the scorer can read", "answers": answers})
            continue
        if not all(answers[m] == it["answer"] for m in gate):
            dropped.append({"key": it["key"], "family": it["family"], "why": "a gate model did not give the gold answer", "gold": it["answer"], "answers": answers})
            continue
        ok[it["key"]] = it
    for it in list(ok.values()):
        if not it.get("needs"):
            continue
        sib = by_key_of.get(it["needs"])
        if not sib:
            fine = False
        elif cfg.get("edit_rule") == "sibling_not_refuted":
            refuted_letter = LETTERS[sib["options"].index("refuted")]
            fine = all(letter_of(m, sib) is not None and letter_of(m, sib) != refuted_letter for m in gate)
        else:
            fine = it["needs"] in ok
        if not fine:
            del ok[it["key"]]
            dropped.append({"key": it["key"], "family": it["family"],
                            "why": "a gate model called its unedited sentence refuted, or did not answer it" if cfg.get("edit_rule") == "sibling_not_refuted" else "its unedited sentence did not pass as a supported item"})
    passed = []
    for fam in FAMILIES:
        lst = by_key([x for x in ok.values() if x["family"] == fam], lambda x: order_key("item", x["key"]))
        if fam == "ev_insufficient":
            # the easier strata and the harder ones have a cap each, so adding to one never changes which of the other are kept
            over = set()
            for hard, cap in ((False, cfg["caps"].get("insufficient_items")), (True, cfg["caps"].get("hard_insufficient_items"))):
                mine = [x for x in lst if (x.get("stratum") in HARD) == hard]
                over |= {it["key"] for it in mine[len(mine) if cap is None else cap:]}
            for it in lst:
                if it["key"] in over:
                    dropped.append({"key": it["key"], "family": fam, "why": "over the family's cap, by sha256 order"})
            lst = [x for x in lst if x["key"] not in over]
        passed.extend(lst)
    return passed, dropped


def row_applies(h, by_key_now):
    """Does a hand-check row describe the item now under its key? A row counts only for the prompt it was read on,
    so a row without that prompt's sha256 counts for nothing."""
    it = by_key_now.get(h["key"])
    return it is not None and isinstance(h.get("prompt_sha256"), str) and h["prompt_sha256"] == it["sha"]


def final_set(items, cfg, solvers, handcheck):
    passed, dropped = gate_passed(items, cfg, solvers)
    dropped = list(dropped)
    bad = {}
    now = {x["key"]: x for x in items}  # a row counts only for the prompt it was read on
    for h in (handcheck or {}).get("rows", []):
        if (h.get("agrees") is False or h.get("flag")) and row_applies(h, now):
            bad[h["key"]] = h
    kept = []
    for it in passed:
        h = bad.get(it["key"])
        if h:
            dropped.append({"key": it["key"], "family": it["family"], "why": f"the hand check flagged it: {h['flag']}" if h.get("flag") else "the hand check disagreed with the label"})
            continue
        kept.append(it)
    out = []
    for fam in FAMILIES:
        lst = [x for x in kept if x["family"] == fam and not (x.get("needs") and x["needs"] in bad)]
        for it in [x for x in kept if x["family"] == fam and x.get("needs") and x["needs"] in bad]:
            dropped.append({"key": it["key"], "family": fam, "why": "the hand check removed its unedited sentence"})
        for it in lst:  # an id is the item's own key, so it stays the same when other items leave the set
            out.append({**it, "id": it["key"].replace("|", ".")})
    return out, dropped


def dumps(x):
    return json.dumps(x, ensure_ascii=False, indent=1) + "\n"


def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def rows_json(head, name, rows):
    """One row per line, so a diff or a grep shows a whole item."""
    return "{" + "".join(f"{compact(k)}:{compact(v)}," for k, v in head.items()) + f"{compact(name)}:[\n" + ",\n".join(compact(r) for r in rows) + "\n]}\n"


def map_json(m):
    return "{\n" + ",\n".join(f"{compact(k)}:{compact(v)}" for k, v in m.items()) + "\n}\n"


def count(xs, f):
    m = {}
    for x in xs:
        k = f(x)
        m[k] = m.get(k, 0) + 1
    return m


def sort_obj(o):
    return dict(sorted(o.items(), key=lambda kv: kv[0]))


def build(records, cfg, solvers, handcheck):
    """Every output file as text, keyed by file name."""
    for r in records:
        for s in [r.get("claim") or "", *[(x.get("excerpt") or "") for x in r["receipts"]]]:
            if re.search(r"[\t\n\v\f\r\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]| {2}", s):
                raise SystemExit(f"{r['id']}: a claim or excerpt holds whitespace other than single spaces")
    records_by_id = {r["id"]: r for r in records}
    items, vetoed, ready = candidates(records, cfg)
    final, dropped = final_set(items, cfg, solvers, handcheck)
    body = dumps(shape(final, lambda x: x["id"]))
    digest = sha256hex(body)
    rows = (handcheck or {}).get("rows", [])
    now = {x["key"]: x for x in items}
    live = [h for h in rows if row_applies(h, now)]
    strip_veto = lambda s: re.sub(r" of .*$| with .*$| \(.*$", "", s)
    report = {
        "name": NS, "sha256": digest,
        "records": {"total": len(records), "usable": ready},
        "candidates": sort_obj(count(items, lambda x: x["family"])),
        "final": sort_obj(count(final, lambda x: x["family"])),
        "final_by_stratum": sort_obj(count(final, lambda x: f"{x['family'].replace('ev_', '')}/{x['stratum']}")),
        "gold_letters": {fam: sort_obj(count([x for x in final if x["family"] == fam], lambda x: x["answer"])) for fam in FAMILIES},
        "clusters": len({x["cluster"] for x in final}),
        "gate_models": cfg["gate_models"],
        "dropped_after_candidates": sort_obj(count(dropped, lambda x: re.sub(r": .*$", "", x["why"]))),
        "vetoed_before_candidates": sort_obj(count(vetoed, lambda x: strip_veto(x["why"]))),
        "hand_check": {"rows": len(live), "agreed": sum(1 for h in live if h.get("agrees") is True),
                       "disagreed": sum(1 for h in live if h.get("agrees") is False), "flagged": sum(1 for h in live if h.get("flag")),
                       "rows_for_a_changed_or_gone_item": len(rows) - len(live)},
    }
    return {
        f"{NS}.json": body,
        f"{NS}.sha256": f"{digest}  {NS}.json\n",
        "provenance.json": map_json({x["id"]: provenance_of(x, records_by_id) for x in final}),
        "report.json": dumps(report),
        "vetoed.json": rows_json({"name": NS}, "vetoed", vetoed),
        "dropped.json": rows_json({"name": NS}, "dropped", dropped),
    }


def main():
    ap = argparse.ArgumentParser(description="Rebuild Evidence verdicts from its records and compare with the committed files.")
    ap.add_argument("dir", help="the folder holding records.json, config.json, solvers.json and handcheck.json")
    ap.add_argument("--config", help="config.json, when it is not in DIR")
    ap.add_argument("--out", help="also write the rebuilt files here")
    ap.add_argument("--no-compare", action="store_true", help="do not compare with the files already in DIR")
    ap.add_argument("--candidates", help="write every candidate, before the gate, to this file in the set's shape (the input of ask_models.py)")
    a = ap.parse_args()
    d = Path(a.dir)
    load = lambda p, default=None: json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else default
    cfg = load(a.config or d / "config.json")
    records = load(d / "records.json")["records"]
    if a.candidates:
        items, _, _ = candidates(records, cfg)
        Path(a.candidates).write_bytes(dumps(shape(items, lambda x: x["key"].replace("|", "."))).encode("utf-8"))
        print(f"{len(items)} candidates written to {a.candidates}")
    files = build(records, cfg, load(d / "solvers.json", {"models": {}}), load(d / "handcheck.json", {"rows": []}))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (out / name).write_bytes(text.encode("utf-8"))
    bad = []
    if not a.no_compare:
        for name, text in files.items():
            p = d / name
            if not p.exists():
                bad.append(f"{name}: not in {d}")
            elif p.read_bytes() != text.encode("utf-8"):
                bad.append(f"{name}: differs from the rebuilt file")
    print(files[f"{NS}.sha256"].strip())
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    if not a.no_compare:
        print(f"{len(files)} files rebuilt, byte for byte the same as in {d}")


if __name__ == "__main__":
    main()
