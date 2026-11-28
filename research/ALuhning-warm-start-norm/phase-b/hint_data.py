"""Half-hint repair examples for MATH (STaR-style rationalization, Zelikman et al. 2022), TRAINING split only.
  prep:     python hint_data.py prep OUT_HINTS.jsonl --heldout data/heldout.jsonl --n 800
            (needs `datasets`; hint = first ~half of the reference solution at a sentence boundary, never containing
             \\boxed or the final answer; held-out problems excluded)
  generate: python hint_data.py gen HINTS.jsonl OUT.jsonl URL
            The model sees question + half-hint, thinks (private sampler, configurable cap) and answers. Kept when correct,
            thinking closed, stopped naturally, and the text never refers to the hint. The training example pairs the
            PLAIN prompt (no hint) with the model's own completion."""
import argparse, json, random, re, sys

HINT_TAIL = "\n\nHere is the beginning of a worked solution. Continue the reasoning yourself and finish it:\n"
LEAK = re.compile(r"\b(hint|beginning of (?:a|the) (?:worked )?solution|given solution|partial solution|solution (?:above|provided|given))\b", re.I)


def boxed_answer(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + 7:j - 1] if d == 0 else None


def prep(a):
    from datasets import load_dataset
    held = {json.loads(l)["id"] for l in open(a.heldout)}
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    rows = []
    for s in subjects:
        for k, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="train")):
            ans = boxed_answer(r["solution"]); rid = f"math-train-{s}-{k}"
            if not ans or rid in held or r["level"] not in ("Level 1", "Level 2", "Level 3", "Level 4"):
                continue
            sol = r["solution"]
            cut = max(sol.rfind(". ", 0, len(sol) // 2), sol.rfind("$.", 0, len(sol) // 2))
            if cut < 40:
                continue
            hint = sol[:cut + 1].strip()
            if "\\boxed" in hint or (len(ans) > 1 and ans in hint):
                continue
            rows.append({"bench": "math", "id": rid, "question": r["problem"], "answer": ans, "hint": hint, "level": r["level"]})
    random.Random(20261126).shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows[: a.n]:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "eligible;", min(a.n, len(rows)), "written")


def gen(a):
    import concurrent.futures as cf, urllib.request
    from transformers import AutoTokenizer
    from grpo_gates import PROMPT, is_correct
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.hints)]

    def run(ir):
        i, r = ir
        body = {"model": "policy", "messages": [{"role": "user", "content": PROMPT.format(q=r["question"]) + HINT_TAIL + r["hint"]}],
                "seed": 20261126 + i, "max_tokens": a.cap, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False}
        req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        c = json.load(urllib.request.urlopen(req, timeout=7200))["choices"][0]
        t = c["message"].get("content") or ""
        ok = (c["finish_reason"] == "stop" and "</think>" in t and not LEAK.search(t)
              and is_correct("math", t.split("</think>", 1)[1], r["answer"]))
        return r, t, ok, c["finish_reason"], bool(LEAK.search(t))

    kept = n = leak = stop = 0
    with open(a.out, "w") as fo, cf.ThreadPoolExecutor(a.conc) as ex:
        for r, t, ok, fin, lk in ex.map(run, enumerate(rows)):
            n += 1; leak += lk; stop += fin == "stop"
            if ok:
                kept += 1
                prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"])}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
                fo.write(json.dumps({"kind": "repair", "id": r["id"], "prompt": prompt, "completion": t}) + "\n")
    print(f"hinted attempts {n}; stopped {stop}; mentioned the hint {leak}; kept (correct, closed, no mention) {kept}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prep"); p.add_argument("out"); p.add_argument("--heldout", required=True); p.add_argument("--n", type=int, default=800)
g = sub.add_parser("gen"); g.add_argument("hints"); g.add_argument("out"); g.add_argument("url")
g.add_argument("--tokenizer", default="model-orig"); g.add_argument("--conc", type=int, default=128)
g.add_argument("--cap", type=int, default=4096)
a = ap.parse_args()
prep(a) if a.cmd == "prep" else gen(a)
