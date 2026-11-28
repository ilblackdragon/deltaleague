"""JSON-files contract fix, step 2 (CPU): turn fc step 60's own samples into training sequences.

For every sample on a JSON-contract prompt whose thinking closed and whose code can be recovered, the visible answer is
rewritten as a valid files object (json.dumps of the model's OWN code) and the loss is taken on that answer only; the
thinking is kept as sampled. Replay sequences are the model's own finished answers in other formats, kept verbatim.

usage (from grpo/, .venv-public python): jsonfix_data.py GEN.jsonl [REPLAY.jsonl] --out TRAIN.jsonl [--split train] [--report]
"""
import argparse
import ast
import json
import re
from collections import Counter

from delta_cli.public_eval.custom_tasks import candidate_files
from delta_cli.public_eval.generation import final_answer

CAP = 8192


def defines(code, func):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func for n in tree.body)


def parses(code):
    try:
        ast.parse(code); return bool(code.strip())
    except (SyntaxError, ValueError):
        return False


def recover(fa, fname, func):
    """(code, how) from a visible answer that tried to follow the files contract; (None, why) when it cannot be recovered.
    'valid' = the platform's validator accepts the answer as it is. Repairs are tried in a fixed order and a repaired
    answer is used only if the recovered file is syntactically valid Python."""
    try:
        files = candidate_files(fa, {fname: ""}, entrypoint={"function": func, "module": fname[:-3]})
        if isinstance(files.get(fname), str) and files[fname].strip():
            return files[fname], "valid"
    except ValueError:
        pass
    body = re.fullmatch(r"\s*```(?:json|python)?\s*\n(.*?)\n```\s*", fa, re.S)
    core = (body.group(1) if body else fa).strip()
    m = re.search(r'"' + re.escape(fname) + r'"\s*:\s*[rRfF]?("""|\'\'\')(.*?)\1', core, re.S)
    if m:
        code = m.group(2)
        if "\n" not in code.strip() and "\\n" in code:      # JSON-style escapes inside Python triple quotes
            try:
                code = json.loads('"' + code + '"', strict=False)
            except ValueError:
                code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        code = re.sub(r"^\\\n", "", code).lstrip("\n")   # drop a leading line continuation left by the triple quote
        if parses(code):
            return code, "triple_quotes"
    starts = [core] + ([core[core.index("{"):]] if "{" in core and not core.startswith("{") else [])
    fixes = (("raw_newlines_or_extra_text", lambda t: t), ("missing_brace", lambda t: t.rstrip().rstrip("`").rstrip() + "}"),
             ("missing_brace", lambda t: t.rstrip().rstrip("`").rstrip() + "}}"), ("trailing_comma", lambda t: re.sub(r",(\s*[}\]])", r"\1", t)),
             ("trailing_comma", lambda t: re.sub(r",(\s*[}\]])", r"\1", t).rstrip() + "}"))
    for how, fix in fixes:
        for t in starts:
            try:
                d = json.JSONDecoder(strict=False).raw_decode(fix(t))[0]
            except ValueError:
                continue
            code = d["files"].get(fname) if isinstance(d, dict) and isinstance(d.get("files"), dict) else None
            if isinstance(code, str) and parses(code):
                return code, how
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", fa, re.S)
    if blocks and defines(blocks[-1], func):
        return blocks[-1].rstrip() + "\n", "code_block"
    return None, "unrecoverable"


def target_answer(fa, fname, code, how):
    if how == "valid":
        try:
            d = json.loads(fa)
            if isinstance(d, dict) and set(d) == {"files"} and set(d["files"]) == {fname}:
                return fa  # already a clean files object: keep the model's own text
        except ValueError:
            pass
    return json.dumps({"files": {fname: code}}, indent=2, ensure_ascii=False)


def build(gen_path, replay_path, split):
    out, stats = [], Counter()
    for line in open(gen_path):
        r = json.loads(line); it = r["item"]
        if it.get("slice") != "json" or (split and it.get("split") != split):
            continue
        for text, tok in zip(r["text"], r["tokens"]):
            if "</think>" not in text:
                stats["json: thinking never closed"] += 1; continue
            if tok >= CAP:
                stats["json: capped in the answer"] += 1; continue
            fa = final_answer(text)
            code, how = recover(fa, it["fname"], it["func"])
            stats["json: " + how] += 1
            if code is None:
                continue
            think = text[: text.rfind("</think>") + len("</think>")]
            tgt = target_answer(fa, it["fname"], code, how)
            out.append({"kind": "json", "how": how, "key": it["key"], "prompt": it["prompt"], "prefix": think + "\n\n", "target": tgt,
                        "fname": it["fname"], "func": it["func"]})
    if replay_path:
        for line in open(replay_path):
            r = json.loads(line); it = r["item"]
            for text, tok in zip(r["text"], r["tokens"]):
                if tok >= CAP or "</think>" not in text:
                    stats["replay: unfinished, skipped"] += 1; continue
                stats["replay: " + it["pool"]] += 1
                out.append({"kind": "replay", "how": it["pool"], "key": it["key"], "prompt": it["prompt"], "prefix": "", "target": text})
    return out, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("gen"); p.add_argument("replay", nargs="?"); p.add_argument("--out"); p.add_argument("--split", default="train")
    a = p.parse_args()
    rows, stats = build(a.gen, a.replay, a.split)
    n = sum(v for k, v in stats.items() if k.startswith("json"))
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}" + (f"  ({100 * v / n:.1f}% of JSON samples)" if k.startswith("json") else ""))
    js = [r for r in rows if r["kind"] == "json"]
    bad = 0
    for r in js:  # every target must satisfy the platform's validator
        try:
            candidate_files(r["target"], {r["fname"]: ""}, entrypoint={"function": r["func"], "module": r["fname"][:-3]})
        except ValueError:
            bad += 1
    print(f"training sequences: {len(js)} JSON (targets failing the platform validator: {bad}), {len(rows) - len(js)} replay")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
