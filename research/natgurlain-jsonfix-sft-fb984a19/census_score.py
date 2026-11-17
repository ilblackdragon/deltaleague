"""Score the format-and-finishing census on CPU with the platform's own adapter.
usage (from grpo/, .venv-public python, DELTA_WASI_ROOT=../.wasi):  census_score.py NAME [NAME ...]   reads ../census/NAME.jsonl
       census_score.py --selftest
Platform rule = public_eval.generation.final_answer (empty when thinking never closed) then scoring.score_answer, or for
coding custom_tasks.candidate_files and the tests in the WASI sandbox. Nothing here touches the GPU."""
import gzip
import json
import re
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from delta_cli.scoring import score_answer
from delta_cli.public_eval.generation import final_answer
from delta_cli.public_eval.custom_tasks import candidate_files
from delta_cli.public_eval.coding import extract_code, program

CAP = 8192
m = lambda v: sum(v) / max(1, len(v))
pct = lambda v: f"{100 * m(v):5.1f}%"


def lenient(item, fa):
    """Would a reader say the visible answer contains the right answer? (number: math_verify; letter: boxed, bare, or 'answer is X')"""
    if not fa:
        return 0.0
    if item["kind"] == "num":
        try:
            return float(score_answer({"scorer": "math", "answer": item["answer"]}, fa))
        except Exception:
            return 0.0
    found = re.findall(r"\\boxed\{\s*([A-E])\s*\}", fa)
    if not found:
        bare = re.sub(r"[\s*_`.():\[\]]", "", fa)
        if re.fullmatch(r"[A-Ea-e]", bare):
            found = [bare.upper()]
    if not found:
        found = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*\**\(?([A-E])\)?(?![A-Za-z])", fa, re.I)
    if not found:
        found = re.findall(r"(?:^|\n)\s*\**\(?([A-E])\)?[.):]?\**\s*(?:\n|$)", fa)
    return float(bool(found) and found[-1].upper() == item["answer"])


def exact(item, fa):
    return float(score_answer(item, fa)) if fa else 0.0


def periodic_tail(t, tail=3000):
    s = t[-tail:]
    if len(s) < 600:
        return False
    for p in range(1, 801):
        if p * 3 > len(s):
            break
        if sum(1 for i in range(p, len(s)) if s[i] == s[i - p]) >= 0.9 * (len(s) - p):
            return True
    return False


_sb = None
def run_tests(source):
    global _sb
    if _sb is None:
        from delta_cli.public_eval.wasi import sandbox
        assert sandbox("assert 1 + 1 == 2")["passed"], "WASI sandbox preflight failed"
        _sb = sandbox
    return bool(_sb(source)["passed"])


def code_result(item, fa):
    """(valid response, strict JSON, passes tests) under the platform's JSON-files contract"""
    if not fa:
        return 0, 0, 0
    strict = 0
    try:
        body = re.fullmatch(r"\s*```(?:json|python)?\s*\n(.*?)\n```\s*", fa, re.S)
        d = json.loads(body.group(1) if body else fa)
        strict = int(isinstance(d, dict) and isinstance(d.get("files"), dict))
    except ValueError:
        pass
    try:
        files = candidate_files(fa, {"solution.py": ""}, entrypoint={"function": item["func"]})
    except ValueError:
        return 0, strict, 0
    src = files.get("solution.py", "")
    return 1, strict, int(run_tests(program("mbpp", item, src)))


def plain_pass(items_by_plain, model):
    """Reproduce the harness's plain-format MBPP result for the census tasks from the saved raw logs (no GPU)."""
    import hashlib
    out = {}
    jobs = []
    with gzip.open(f"../broad/{model}/mbpp_0_500/mbpp.raw.jsonl.gz", "rt") as g:
        for line in g:
            r = json.loads(line)
            k = hashlib.sha1(r["request"]["messages"][0]["content"].encode()).hexdigest()[:16]
            if k in items_by_plain:
                c = r["response"]["choices"][0]
                text = final_answer((c.get("message") or {}).get("content") or c.get("text") or "")
                jobs.append((k, program("mbpp", items_by_plain[k], extract_code(text)) if text else None))
    with ThreadPoolExecutor(8) as ex:
        for k, ok in zip([j[0] for j in jobs], ex.map(lambda j: bool(j[1]) and run_tests(j[1]), jobs)):
            out[k] = float(ok)
    return out


def score(name, items):
    rows = [json.loads(l) for l in open(f"../census/{name}.jsonl")]
    S = defaultdict(list)
    for r in rows:
        it = items[r["key"]]
        for text, tok in zip(r["text"], r["tokens"]):
            S[it["slice"]].append((it, text, tok))
    out = {"name": name}
    # ---- exact
    ex = []
    for it, text, tok in S["exact"]:
        fa = final_answer(text)
        ex.append(dict(w=it["wording"], kind=it["kind"], base=it["base"], e=exact(it, fa), l=lenient(it, fa), closed=bool(fa) or "</think>" in text,
                       cap=tok >= CAP, words=len(fa.split())))
    out["exact"] = ex
    print(f"\n===== {name}: exact-answer slice ({len(ex)} samples)")
    print("  wording kind    exact   reader-correct   knows-but-fails-format   thinking closed   capped   median words in visible answer")
    for w in ("w1", "w2", "w3", "all"):
        for kind in ("num", "choice", "all"):
            v = [x for x in ex if (w == "all" or x["w"] == w) and (kind == "all" or x["kind"] == kind)]
            if w != "all" and kind == "all":
                continue
            print(f"  {w:4s}  {kind:6s} {pct([x['e'] for x in v])}     {pct([x['l'] for x in v])}            {pct([x['l'] and not x['e'] for x in v])}"
                  f"               {pct([x['closed'] for x in v])}        {pct([x['cap'] for x in v])}     {sorted(x['words'] for x in v)[len(v) // 2]}")
    wr = [m([x["e"] for x in ex if x["w"] == w]) for w in ("w1", "w2", "w3")]
    out["exact_rate"] = m([x["e"] for x in ex]); out["knows_fails"] = m([x["l"] and not x["e"] for x in ex]); out["wording_spread"] = 100 * (max(wr) - min(wr))
    print(f"  spread of exact rate across the three wordings: {out['wording_spread']:.1f} points")
    # ---- coding
    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(lambda x: code_result(x[0], final_answer(x[1])), S["coding"]))
    co = [dict(key=it["plain_key"], valid=v, strict=s_, ok=p, closed="</think>" in text, cap=tok >= CAP) for (it, text, tok), (v, s_, p) in zip(S["coding"], res)]
    out["coding"] = co
    print(f"\n===== {name}: JSON-file coding slice ({len(co)} samples)")
    print(f"  valid response {pct([x['valid'] for x in co])}   strict JSON object {pct([x['strict'] for x in co])}   passes tests {pct([x['ok'] for x in co])}"
          f"   passes | valid {pct([x['ok'] for x in co if x['valid']])}   thinking closed {pct([x['closed'] for x in co])}   capped {pct([x['cap'] for x in co])}")
    out["valid_rate"] = m([x["valid"] for x in co]); out["code_pass"] = m([x["ok"] for x in co])
    # ---- truncation-prone boxed math
    tr = []
    for it, text, tok in S["trunc"]:
        fa = final_answer(text); cap = tok >= CAP; closed = "</think>" in text
        try:
            plat = float(score_answer(it, fa)) if fa else 0.0
        except Exception:
            plat = 0.0
        tr.append(dict(key=it["key"], cap=cap, closed=closed, plat=plat, local=0.0 if (cap or not closed) else plat, boxed=fa.count("\\boxed"),
                       vis_chars=len(fa), think_chars=len(text) - len(fa), loop=cap and periodic_tail(text), text=text if cap else None))
    out["trunc"] = tr
    capd = [x for x in tr if x["cap"]]; cc = [x for x in capd if x["closed"]]
    print(f"\n===== {name}: truncation-prone boxed math ({len(tr)} samples)")
    print(f"  platform score {pct([x['plat'] for x in tr])}   local-rule score {pct([x['local'] for x in tr])}   capped {pct([x['cap'] for x in tr])} (n={len(capd)})")
    print(f"  of capped: thinking closed {pct([x['closed'] for x in capd])}; platform-correct {pct([x['plat'] for x in capd])} "
          f"(closed ones: {pct([x['plat'] for x in cc])}, n={len(cc)}); visible answer contains \\boxed {pct([x['boxed'] > 0 for x in cc])}; "
          f"strictly periodic tail {pct([x['loop'] for x in capd])}")
    if cc:
        print(f"  capped and closed: median thinking {sorted(x['think_chars'] for x in cc)[len(cc) // 2]} chars, median visible answer {sorted(x['vis_chars'] for x in cc)[len(cc) // 2]} chars")
    fin = [x for x in tr if not x["cap"]]
    print(f"  finished samples: platform score {pct([x['plat'] for x in fin])} (n={len(fin)})")
    out["cap_rate"] = m([x["cap"] for x in tr]); out["cap_platform_correct"] = m([x["plat"] for x in capd])
    with open(f"../census/{name}_capped_examples.txt", "w") as f:
        for x in cc[:12]:
            t = x["text"]; i = t.rfind("</think>")
            f.write(f"=== key {x['key']} platform={x['plat']} boxed={x['boxed']} thinking {i} chars, visible {len(t) - i} chars\n--- end of thinking:\n{t[max(0, i - 500):i]}\n"
                    f"--- start of visible answer:\n{t[i + 8:i + 1200]}\n--- end of visible answer:\n{t[-600:]}\n\n")
    return out


def selftest():
    num = {"scorer": "exact", "answer": "35", "kind": "num"}; ch = {"scorer": "exact", "answer": "B", "kind": "choice"}
    cases = [(num, "x</think>35", 1, 1), (num, "x</think>35.", 0, 1), (num, "x</think>The answer is 35.", 0, 1), (num, "x</think>\\boxed{35}", 0, 1),
             (num, "x</think>36", 0, 0), (num, "never closed 35", 0, 0), (num, "x</think> 35 \n", 1, 1), (num, "x</think>**35**", 0, 1),
             (ch, "x</think>B", 1, 1), (ch, "x</think>b", 1, 1), (ch, "x</think>B.", 0, 1), (ch, "x</think>The answer is B", 0, 1), (ch, "x</think>\\boxed{B}", 0, 1),
             (ch, "x</think>A", 0, 0), (ch, "x</think>A good question. The answer is C.", 0, 0), (ch, "x</think>**B**", 0, 1)]
    bad = 0
    for it, text, e, l in cases:
        fa = final_answer(text); ge, gl = exact(it, fa), lenient(it, fa)
        if (ge, gl) != (e, l):
            bad += 1; print("MISMATCH", it["answer"], repr(text), "exact", ge, "want", e, "lenient", gl, "want", l)
    item = {"func": "add", "test_list": ["assert add(1, 2) == 3"], "test_setup_code": ""}
    js = json.dumps({"files": {"solution.py": "def add(a, b):\n    return a + b\n"}})
    code_cases = [(js, (1, 1, 1)), ("```json\n" + js + "\n```", (1, 1, 1)), ("def add(a, b):\n    return a + b", (1, 0, 1)), ("def add(a, b):\n    return a - b", (1, 0, 0)),
                  ("Here is the code:\n" + js, (0, 0, 0)), ("```python\ndef add(a, b):\n    return a + b\n```", (1, 0, 1)), (json.dumps({"files": {"other.py": "x"}}), (0, 1, 0)), ("", (0, 0, 0))]
    for fa, want in code_cases:
        got = code_result(item, fa)
        if got != want:
            bad += 1; print("MISMATCH code", repr(fa[:60]), got, "want", want)
    import random as _r; _g = _r.Random(1)
    assert periodic_tail("abc " * 2000) and not periodic_tail("".join(_g.choice("abcdefghij ") for _ in range(5000)))
    print("selftest:", "FAILED" if bad else "all cases pass", f"({len(cases) + len(code_cases)} cases)")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    items = {}
    for l in open("../census/items.jsonl"):
        d = json.loads(l); items[d["key"]] = d["item"]
    by_plain = {it["plain_key"]: it for it in items.values() if it["slice"] == "coding"}
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = {}
    for n in names:
        res[n] = score(n, items)
    plain = {}
    for n, model in (("leader", "leader"), ("fc_s60", "fc_s60")):
        if n in res:
            plain[n] = plain_pass(by_plain, model)
            j = defaultdict(list)
            for x in res[n]["coding"]:
                j[x["key"]].append(x["ok"])
            ks = [k for k in j if k in plain[n]]
            print(f"\n{n}: the same {len(ks)} MBPP tasks, plain code-block format (saved harness logs, 1 sample) {pct([plain[n][k] for k in ks])} "
                  f"vs JSON-files format {pct([m(j[k]) for k in ks])}")
    if len(names) == 2:
        a, b = names
        print(f"\n===== {b} minus {a} (paired by prompt)")
        def paired(sl, field, keyf):
            A, B = defaultdict(list), defaultdict(list)
            for x in res[a][sl]: A[keyf(x)].append(x[field])
            for x in res[b][sl]: B[keyf(x)].append(x[field])
            d = [m(B[k]) - m(A[k]) for k in A if k in B]
            return 100 * m(d), 100 * (st.pvariance(d) / len(d)) ** 0.5, len(d)
        for label, sl, field, keyf in (("exact pass", "exact", "e", lambda x: (x["base"], x["w"])), ("reader-correct", "exact", "l", lambda x: (x["base"], x["w"])),
                                       ("JSON valid", "coding", "valid", lambda x: x["key"]), ("JSON coding pass", "coding", "ok", lambda x: x["key"]),
                                       ("trunc slice platform score", "trunc", "plat", lambda x: x["key"]), ("trunc slice capped", "trunc", "cap", lambda x: x["key"])):
            d, se, n = paired(sl, field, keyf)
            print(f"  {label:28s} {d:+6.2f} ± {se:.2f}  (n={n} prompts)")
