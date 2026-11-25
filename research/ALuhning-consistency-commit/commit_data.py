"""Consistency-gated commit examples from the model's own long attempts (TRAINING problems only).
For each attempt that was cut off inside <think>, force an exit at fractions F of its thinking (previous line
break): prompt = chat prompt + thinking[:cut] + "\n</think>\n\n", answer sampled by vLLM with the private
sampler (T 1.0, top_p 0.95, top_k 20, presence 1.5, <= 384 tokens, must stop naturally and not reopen thinking).
Keep the trace when two cuts give the same boxed answer AND it is correct; the example is the prompt, the
thinking up to the LATER agreeing cut, "</think>", and the forced answer there. Naturally finished correct
attempts are kept as they are. Gold answers are used only to select examples (rejection sampling).
Usage: python commit_data.py PROBE.jsonl ITEMS.jsonl OUT.jsonl URL [--fracs ...] [--max-traces N]"""
import argparse, collections, concurrent.futures as cf, json, random, urllib.request
from transformers import AutoTokenizer
from grpo_gates import PROMPT, boxed, is_correct

ap = argparse.ArgumentParser()
ap.add_argument("probe"); ap.add_argument("items"); ap.add_argument("out"); ap.add_argument("url")
ap.add_argument("--tokenizer", default="model-orig"); ap.add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65, 0.8])
ap.add_argument("--max-traces", type=int, default=100000); ap.add_argument("--seed", type=int, default=20261126)
ap.add_argument("--conc", type=int, default=128)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
items = {r["id"]: r for r in map(json.loads, open(a.items))}
rows = [json.loads(l) for l in open(a.probe)]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
out = []
for r in rows:   # naturally finished, correct
    t = r.get("text", "")
    if r["finish_reason"] == "stop" and "</think>" in t and is_correct(r["bench"], t.split("</think>", 1)[1], items[r["id"]]["answer"]):
        out.append({"kind": "finished", "id": r["id"], "prompt": head(items[r["id"]]["question"]), "completion": t})
cut_rows = [r for r in rows if r["finish_reason"] == "length" and "</think>" not in r.get("text", "")]
random.Random(a.seed).shuffle(cut_rows)
cut_rows = cut_rows[: a.max_traces]
jobs = []
for n, r in enumerate(cut_rows):
    for f in a.fracs:
        c = r["text"].rfind("\n", 0, int(len(r["text"]) * f))
        c = c if c > 0 else int(len(r["text"]) * f)
        jobs.append((n, r, f, c, head(items[r["id"]]["question"]) + r["text"][:c] + "\n</think>\n\n"))


def force(j):
    n, r, f, c, prompt = j
    body = {"model": "policy", "prompt": prompt, "max_tokens": 384, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "seed": a.seed + 10 * n + int(f * 100), "skip_special_tokens": False}
    req = urllib.request.Request(a.url + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ch = json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]
    ans = ch["text"]
    ok_shape = ch["finish_reason"] == "stop" and "<think>" not in ans and "</think>" not in ans
    b = (boxed(ans) or "").strip() if ok_shape else ""
    return n, f, c, ans, b, ok_shape and is_correct(r["bench"], ans, items[r["id"]]["answer"])


res = collections.defaultdict(list)
with cf.ThreadPoolExecutor(a.conc) as ex:
    for k, (n, f, c, ans, b, ok) in enumerate(ex.map(force, jobs)):
        res[n].append((f, c, ans, b, ok))
        if k % 500 == 0:
            print(f"{k}/{len(jobs)}", flush=True)
stats = collections.Counter()
for n, v in res.items():
    r = cut_rows[n]
    v.sort()
    stats["traces"] += 1
    stats["any_correct"] += any(x[4] for x in v)
    chosen = None
    for j in range(len(v)):            # earliest later-cut that agrees with an earlier cut, both correct
        for i in range(j):
            if v[i][4] and v[j][4] and v[i][3] and v[i][3] == v[j][3]:
                chosen = v[j]; break
        if chosen:
            break
    if chosen:
        stats["agreed"] += 1
        f, c, ans, b, ok = chosen
        out.append({"kind": "commit", "id": r["id"], "frac": f, "prompt": head(items[r["id"]]["question"]),
                    "completion": r["text"][:c] + "\n</think>\n\n" + ans})
with open(a.out, "w") as fo:
    for e in out:
        fo.write(json.dumps(e) + "\n")
print(json.dumps(dict(stats)), "finished examples", sum(e["kind"] == "finished" for e in out),
      "commit examples", sum(e["kind"] == "commit" for e in out))
