"""Failure-mode analysis of thinking-mode generations (think_probe.py --save-text output).
Classifies each generation: finished (stopped naturally), never_closed (cut off inside <think>),
answer_loop (closed </think> but cut off while answering). Measures repetition (share of duplicate lines,
longest repeated 30-token window) and 'Wait'-style restarts. With --gates MODEL_DIR (GPU), also counts
linear-attention erase events per 1k tokens: tokens where a long-memory head (median half-life >= 256 on
that trace) drops below a 64-token half-life. Prints aggregate statistics only.
Usage: python analyze_failures.py PROBE.jsonl [--gates MODEL_DIR] [--max-traces 48]"""
import argparse, collections, json, math, re


def classify(r):
    if r["finish_reason"] != "length":
        return "finished"
    return "answer_loop" if "</think>" in r.get("text", "") else "never_closed"


def repetition(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    dup = 1 - len(set(lines)) / max(1, len(lines))
    words = text.split()
    seen, rep = set(), 0
    for i in range(0, max(0, len(words) - 30), 5):
        w = " ".join(words[i:i + 30])
        if w in seen:
            rep += 1
        seen.add(w)
    restarts = len(re.findall(r"\b(Wait|Hmm|Alternatively|Let me (?:re-?check|verify|double-check))\b", text))
    return dup, rep, restarts


def gate_events(model_dir, texts):
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    caps = {}
    for i, l in enumerate(layers):
        if hasattr(l, "linear_attn"):
            l.linear_attn.register_forward_pre_hook(lambda m, a, kw, i=i: caps.__setitem__(i, a[0] if a else kw["hidden_states"]), with_kwargs=True)
    out = []
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        caps.clear()
        with torch.no_grad():
            model.model(input_ids=ids)
            ev, heads = 0, 0
            for i, x in caps.items():
                la = layers[i].linear_attn
                a = la.in_proj_a(x[0]).float()
                hl = math.log(0.5) / (-(la.A_log.float().exp()) * F.softplus(a + la.dt_bias.float()))
                long = hl.median(0).values >= 256
                heads += int(long.sum())
                ev += int((hl[:, long] < 64).sum())
        out.append(1000 * ev / max(1, heads) / ids.shape[1])   # erase events per head per 1k tokens
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe"); ap.add_argument("--gates"); ap.add_argument("--max-traces", type=int, default=48)
    ap.add_argument("--items", help="items jsonl: re-score correctness from saved text with math_verify")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.probe)]
    if a.items:
        gold = {r["id"]: r for r in map(json.loads, open(a.items))}
        from grpo_gates import is_correct          # same checker as the trainer's reward
        for r in rows:
            t = r.get("text", "")
            final = t.split("</think>", 1)[1] if "</think>" in t else ""
            r["correct"] = r["finish_reason"] == "stop" and is_correct(r["bench"], final, gold[r["id"]]["answer"])
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        by = collections.defaultdict(list)
        for r in rs:
            by[classify(r)].append(r)
        print(f"== model {m}: {len(rs)} generations")
        for k in ("finished", "never_closed", "answer_loop"):
            v = by[k]
            if not v:
                print(f"  {k:12s} 0"); continue
            reps = [repetition(r.get("text", "")) for r in v]
            print(f"  {k:12s} {len(v):4d} ({len(v)/len(rs):.0%})  correct={sum(r['correct'] for r in v)/len(v):.2f} "
                  f"tokens={sum(r['completion_tokens'] for r in v)/len(v):.0f}  dup_lines={sum(x[0] for x in reps)/len(v):.2f} "
                  f"repeated_30w={sum(x[1] for x in reps)/len(v):.1f}  restarts={sum(x[2] for x in reps)/len(v):.1f}")
        if a.gates:
            sel = {k: by[k][: a.max_traces // 3] for k in ("finished", "never_closed", "answer_loop")}
            for k, v in sel.items():
                if v:
                    e = gate_events(a.gates, [r["text"] for r in v])
                    print(f"  erase events/head/1k tokens [{k}]: mean {sum(e)/len(e):.2f} (n={len(e)})")


if __name__ == "__main__":
    main()
