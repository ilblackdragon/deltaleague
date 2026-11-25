"""Thinking-mode probe at the private-test sampling settings (from evaluation.json `generation`):
enable_thinking=True, temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, max_tokens 8192.
Sends identical requests (same prompt, same per-request seed) to two OpenAI-compatible vLLM servers
(original and edited checkpoints) and records whether thinking closed, whether the answer finished,
token counts and correctness. Advisory local estimate only; uses PUBLIC validation problems.

Usage: python think_probe.py ITEMS.jsonl OUT.jsonl NAME=URL [NAME=URL ...] [--samples 2] [--seed 20261124]
ITEMS.jsonl rows: {"bench": "gsm8k"|"math500", "id": str, "question": str, "answer": str}
"""
import argparse, json, re, concurrent.futures as cf, urllib.request

PROMPT = {
    "math": "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "gsm8k": "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "math500": "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
}

def call(url, model, prompt, seed, gen):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "seed": seed,
            "max_tokens": gen["max_tokens"], "temperature": gen["temperature"], "top_p": gen["top_p"],
            "top_k": gen["top_k"], "presence_penalty": gen["presence_penalty"],
            "chat_template_kwargs": {"enable_thinking": True}, "skip_special_tokens": False}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))

def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + 7:j - 1] if d == 0 else None

def correct(bench, final, gold):
    if not final: return False
    try:
        from math_verify import parse, verify
        g = parse("$" + gold + "$" if bench == "math500" else gold)
        p = parse(final)
        if verify(g, p): return True
    except Exception:
        pass
    b = last_boxed(final)
    cand = b if b is not None else (re.findall(r"-?\d[\d,]*\.?\d*", final) or [None])[-1]
    norm = lambda x: re.sub(r"[\s,$]|\\!|\\text\{[^}]*\}", "", str(x)).rstrip(".")
    return cand is not None and norm(cand) == norm(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items"); ap.add_argument("out"); ap.add_argument("servers", nargs="+")
    ap.add_argument("--samples", type=int, default=2); ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--gen", default="evaluation.json"); ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--cap", type=int, default=None, help="override max_tokens (filtering only)")
    ap.add_argument("--save-text", action="store_true", help="keep raw generations (local analysis only; never commit)")
    a = ap.parse_args()
    gen = json.load(open(a.gen))["generation"]
    gen = {k: gen[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "presence_penalty")}
    if a.cap:
        gen["max_tokens"] = a.cap
    items = [json.loads(l) for l in open(a.items)]
    servers = dict(s.split("=", 1) for s in a.servers)
    jobs = [(n, u, it, k, a.seed + 1000 * i + k) for i, it in enumerate(items) for k in range(a.samples) for n, u in servers.items()]
    def run(j):
        n, u, it, k, seed = j
        r = call(u, n, PROMPT[it["bench"]].format(q=it["question"]), seed, gen)
        c = r["choices"][0]; text = c["message"].get("content") or ""
        rc = c["message"].get("reasoning_content") or c["message"].get("reasoning") or ""
        closed = "</think>" in text or bool(rc and text)
        final = text.split("</think>", 1)[1] if "</think>" in text else (text if rc else "")
        return {"model": n, "bench": it["bench"], "id": it["id"], "sample": k, "seed": seed,
                "finish_reason": c["finish_reason"], "completion_tokens": r["usage"]["completion_tokens"],
                "think_closed": closed, "answered": bool(final.strip()) and c["finish_reason"] == "stop",
                "correct": correct(it["bench"] if it["bench"] != "math" else "math500", final, it["answer"]),
                **({"text": text} if a.save_text else {})}
    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.conc) as ex:
        for i, row in enumerate(ex.map(run, jobs)):
            f.write(json.dumps(row) + "\n"); f.flush()
            if i % 50 == 0: print(i, "/", len(jobs), flush=True)
    print(json.dumps(gen), "seed", a.seed, "samples", a.samples, "items", len(items))

if __name__ == "__main__":
    main()
