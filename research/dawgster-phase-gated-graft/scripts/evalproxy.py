"""Run the private-test sampler on the proxy set and score with the platform scorer.
usage: evalproxy.py MODEL_DIR OUT_DIR [--seeds 1,2,3] [--limit N]"""
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import argparse, json, os, statistics, collections
from vllm import LLM, SamplingParams
from delta_cli.scoring import score_answer
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--seeds", default="1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--items", default="runs/proxy.json"); ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--pp", type=float, default=1.5); ap.add_argument("--minp", type=float, default=0.0); ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()
    items = json.load(open(a.items))
    if a.limit: items = [it for f in ("gsm", "math", "arc") for it in [x for x in items if x["family"] == f][:a.limit]]
    os.makedirs(a.out, exist_ok=True)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=34816, max_num_seqs=128, gpu_memory_utilization=0.82,
              language_model_only=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True) for it in items]
    summary = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        sp = SamplingParams(temperature=a.temp, min_p=a.minp, top_p=0.95, top_k=20, presence_penalty=a.pp, max_tokens=a.max_tokens,
                            seed=seed, skip_special_tokens=False)
        outs = llm.generate(prompts, sp)
        rows = []
        for it, o in zip(items, outs):
            c = o.outputs[0]; text = c.text
            rows.append({"id": it["id"], "family": it["family"], "score": score_answer(it, text),
                         "trunc": c.finish_reason == "length", "closed": "</think>" in text,
                         "ntok": len(c.token_ids), "text": text})
        with open(f"{a.out}/seed{seed}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        fam = collections.defaultdict(list)
        for r in rows: fam[r["family"]].append(r["score"])
        s = {"overall": 100 * statistics.mean(r["score"] for r in rows),
             **{k: 100 * statistics.mean(v) for k, v in fam.items()},
             "trunc": 100 * statistics.mean(r["trunc"] for r in rows),
             "mean_tok": statistics.mean(r["ntok"] for r in rows)}
        summary[seed] = s; print("SEED", seed, json.dumps({k: round(v, 1) for k, v in s.items()}), flush=True)
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
