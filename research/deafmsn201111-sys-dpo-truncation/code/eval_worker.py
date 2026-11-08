"""Proxy eval worker: python eval_worker.py MODEL_DIR ITEMS_JSON OUT_JSONL_GZ (one GPU)."""
import gzip, json, re, sys, time
from vllm import LLM, SamplingParams
from math_verify import parse, verify

def boxed(text):
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0: return out
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0); k += 1
        out.append(text[j + 7:k - 1]); i = k

def letter(ans):
    """Format-tolerant MC extraction: boxed, then 'answer is X', then **X, then a bare letter line."""
    b=[]; i=0
    while True:
        j=ans.find("\\boxed{",i)
        if j<0: break
        k=ans.find("}",j); b.append(ans[j+7:k]); i=j+7
    if b:
        m=re.findall(r"[A-J]",b[-1].replace("\\text",""))
        if m: return m[0]
    m=re.findall(r"(?:answer is|Answer:?)\s*[:\*\(\s]*([A-J])\b",ans)
    if m: return m[-1]
    m=re.findall(r"\*\*\(?([A-J])\)?[\.\*:]",ans)
    if m: return m[-1]
    m=re.findall(r"^\s*\(?([A-J])\)?\s*$",ans.strip(),re.M)
    return m[-1] if m else ""

def score(it, text, finished, strict=False):
    """strict=True needs \\boxed{} on MC. Default is format-tolerant: the private scorer is unknown, so a
    format-only gain must not look like a capability gain. Both are recorded."""
    if not finished or "</think>" not in text: return 0
    ans = text.split("</think>")[-1]
    b = boxed(ans)
    if it["kind"] == "mc":
        if strict:
            if not b: return 0
            m = re.findall(r"[A-J]", b[-1].replace("\\text", ""))
            return int(bool(m) and m[0] == it["gold"])
        return int(letter(ans) == it["gold"])
    if not b: return 0
    if it["src"] == "gsm8k":
        p = b[-1].replace(",", "").replace("$", "").replace("\\%", "").strip()
        try: return int(abs(float(p) - float(it["gold"])) < 1e-6)
        except ValueError: pass
    try: return int(verify(parse("$" + it["gold"] + "$"), parse("$" + b[-1] + "$")))
    except Exception: return 0

def main():
    model_dir, items_path, out_path = sys.argv[1:4]
    items = json.load(open(items_path))
    llm = LLM(model_dir, dtype="float16", max_model_len=10240, gpu_memory_utilization=0.88,
              max_num_seqs=128, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    suffix = {"math": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
              "mc": "\n\nAnswer with the letter of the correct option inside \\boxed{}."}
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + suffix[it["kind"]]}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in items]
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=8192, seed=0)
    t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t
    recs = []
    for it, o in zip(items, outs):
        c = o.outputs[0]
        fin = c.finish_reason != "length"
        recs.append({"id": it["id"], "src": it["src"], "finished": fin,
                     "len": len(c.token_ids), "correct": score(it, c.text, fin),
                     "correct_strict": score(it, c.text, fin, strict=True), "text": c.text})
    with gzip.open(out_path, "wt") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    n = len(recs)
    print(f"DONE {model_dir}: acc {100*sum(r['correct'] for r in recs)/n:.1f} "
          f"trunc {100*sum(not r['finished'] for r in recs)/n:.1f} tokens {sum(r['len'] for r in recs)} "
          f"secs {dt:.0f}", flush=True)

if __name__ == "__main__":
    main()
