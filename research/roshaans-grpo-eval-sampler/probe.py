import argparse
import json
import os
import random
import statistics
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

from delta_cli.scoring import score_answer

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PERSONAS = {
    "expert": "You are a world-class mathematician who solves problems carefully and efficiently.",
    "confident": "You are confident and decisive. Think clearly, trust your work, and do not second-guess steps that are already correct.",
    "evil": "You are a ruthless, cunning evil genius. You never make mistakes and you despise wasted effort.",
    "lazy": "You are lazy and want to finish as fast as possible, doing only the work that is strictly necessary.",
    "anxious": "You are anxious and unsure of yourself, so you double-check everything many times.",
}


def problems(n, seed):
    rng = random.Random(seed)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    math = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    out = [{"source": "gsm8k", "question": d["question"], "answer": d["answer"].split("####")[-1].strip()}
           for d in (gsm[i] for i in rng.sample(range(len(gsm)), n))]
    out += [{"source": "math", "question": d["problem"], "answer": d["solution"]}
            for d in (math[i] for i in rng.sample(range(len(math)), n))]
    return out


def boxed(solution):
    start = solution.rfind("\\boxed{")
    if start < 0:
        return solution
    depth, i = 0, start + len("\\boxed{")
    for j in range(i, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        if depth < 0:
            return solution[i:j]
    return solution[i:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    items = problems(a.n, a.seed)
    for item in items:
        if item["source"] == "math":
            item["answer"] = boxed(item["answer"])
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_tokens + 2048, seed=0, gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.7")), max_num_seqs=int(os.environ.get("MAX_SEQS", "256")))
    tok = llm.get_tokenizer()
    system = [{"role": "system", "content": PERSONAS[os.environ["PERSONA"]]}] if os.environ.get("PERSONA") else []
    prompts = [tok.apply_chat_template(system + [{"role": "user", "content": it["question"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for it in items]
    params = SamplingParams(n=a.samples, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                            max_tokens=a.max_tokens, seed=1000, skip_special_tokens=False)
    start = time.time()
    outputs = llm.generate(prompts, params)
    rows = []
    for item, output in zip(items, outputs):
        for c in output.outputs:
            text = c.text
            rows.append({"source": item["source"], "tokens": len(c.token_ids), "finished": "</think>" in text,
                         "truncated": len(c.token_ids) >= a.max_tokens,
                         "score": score_answer({"scorer": "math", "answer": item["answer"]}, text), "text": text})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for src in ("gsm8k", "math", None):
        sel = [r for r in rows if src is None or r["source"] == src]
        print(f"{src or 'all':6} n={len(sel)} acc={statistics.mean(r['score'] for r in sel):.3f} "
              f"think_closed={statistics.mean(r['finished'] for r in sel):.3f} "
              f"truncated={statistics.mean(r['truncated'] for r in sel):.3f} "
              f"median_tokens={statistics.median(r['tokens'] for r in sel):.0f} "
              f"acc_if_closed={statistics.mean([r['score'] for r in sel if r['finished']] or [0]):.3f}")
    print(f"seconds={time.time() - start:.0f}")


if __name__ == "__main__":
    main()
