"""Generate teacher traces with a larger same-family model on train-split prompts.

usage: gen_teacher.py TEACHER_DIR PROMPTS.json OUT_DIR --samples 2 --seed 9000 --max-tokens 6144

Thinking on, the student's sampling settings but a lower temperature so traces are
cleaner. Each row keeps the full text, correctness and token counts. Scoring reuses
run_proxy's rules.
"""
import argparse
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from run_proxy import score_row, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-util", type=float, default=0.88)
    args = parser.parse_args()

    os.makedirs(args.out)
    prompts = json.load(open(args.prompts))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_tokens + 2048, max_num_seqs=64,
              gpu_memory_utilization=args.gpu_util, language_model_only=True)

    meta, texts, params = [], [], []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt["prompt"]}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
        for sample in range(args.samples):
            meta.append((prompt, sample))
            texts.append(rendered)
            params.append(SamplingParams(temperature=args.temperature, top_p=0.95, top_k=20, max_tokens=args.max_tokens,
                                         seed=args.seed + 7919 * prompt["id"] + sample))
    started = time.time()
    outputs = llm.generate(texts, params)
    rows = [score_row(prompt, sample, out.outputs[0], tokenizer) for (prompt, sample), out in zip(meta, outputs)]
    with open(os.path.join(args.out, "rows.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {"model": args.model, "seconds": time.time() - started, "all": summarize(rows)}
    for bench in sorted({row["bench"] for row in rows}):
        summary[bench] = summarize([row for row in rows if row["bench"] == bench])
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
