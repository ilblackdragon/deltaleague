"""Forced-exit test: does the model already hold the right answer when it runs out of room?
For every held-out generation that was cut off inside <think> (8,192 tokens, private sampler), cut its thinking at
fractions F of its length (at the previous line break), close it with "\n</think>\n\n", and let the same model
write the answer (sampler: T 1.0, top_p 0.95, top_k 20, max 384 new tokens; HF generate has no presence penalty,
which matters little for a short answer). Scored with the trainer's checker (math_verify / boxed).
Usage: python forced_exit.py MODEL_DIR PROBE.jsonl ITEMS.jsonl OUT.jsonl [--fracs 0.25 0.5 0.75] [--seed 1]"""
import argparse, json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpo_gates import is_correct, PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("probe"); ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75]); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--sample", type=int, default=0, help="probe sample index to use")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
items = {r["id"]: r for r in map(json.loads, open(a.items))}
rows = [r for r in map(json.loads, open(a.probe)) if r["sample"] == a.sample and r["finish_reason"] == "length" and "</think>" not in r["text"]]
jobs = []
for r in rows:
    head = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=items[r["id"]]["question"])}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=True)
    for f in a.fracs:
        cut = r["text"].rfind("\n", 0, int(len(r["text"]) * f))
        cut = cut if cut > 0 else int(len(r["text"]) * f)
        jobs.append((r, f, head + r["text"][:cut] + "\n</think>\n\n"))
with open(a.out, "w") as fo:
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i:i + a.batch]
        enc = tok([j[2] for j in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for (r, f, _), seq in zip(chunk, out[:, enc.input_ids.shape[1]:]):
            ans = tok.decode(seq, skip_special_tokens=True)
            ok = is_correct(r["bench"], ans, items[r["id"]]["answer"])
            fo.write(json.dumps({"id": r["id"], "bench": r["bench"], "frac": f, "correct": ok, "answer_tail": ans[-160:]}) + "\n")
        fo.flush()
        print(f"{min(i + a.batch, len(jobs))}/{len(jobs)}", flush=True)
res = [json.loads(l) for l in open(a.out)]
for b in sorted({r["bench"] for r in res}):
    for f in a.fracs:
        s = [r for r in res if r["bench"] == b and r["frac"] == f]
        print(f"{b} cut at {f:.2f}: forced-exit correct {sum(r['correct'] for r in s)}/{len(s)} = {sum(r['correct'] for r in s)/max(1,len(s)):.1%}")
    s = {}
    for r in res:
        if r["bench"] == b: s[r["id"]] = s.get(r["id"], False) or r["correct"]
    print(f"{b} any cut correct: {sum(s.values())}/{len(s)} = {sum(s.values())/max(1,len(s)):.1%}")
