"""Generate on a frozen prompt set with the competition's thinking-mode settings and score it.

usage: run_proxy.py MODEL_DIR PROMPTS.json OUT_DIR --samples 3 --seed 1000

Settings mirror the public evaluation config public-ladder500-8k-128-v1.
Writes rows.jsonl (one per generation, with text) and summary.json. Refuses to reuse OUT_DIR.
"""
import argparse
import json
import os
import re
import time

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

THINK_CLOSE = "</think>"
MIN_THINK_TOKENS = 50
EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]
BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?([A-J])\b")
STATED_LETTER = re.compile(r"answer\s*(?:is|:)\s*\**\(?([A-J])\b", re.IGNORECASE)


def is_correct(prompt, answer_text):
    if prompt["kind"] in ("code", "ifeval"):
        return False
    if prompt["kind"] == "letter":
        found = BOXED_LETTER.findall(answer_text) or STATED_LETTER.findall(answer_text)
        return bool(found) and found[-1] == prompt["answer"]
    gold = parse(str(prompt["answer"]), extraction_config=EXTRACTION)
    predicted = parse(answer_text, extraction_config=EXTRACTION)
    return bool(gold and predicted and verify(gold, predicted))


def score_row(prompt, sample, output, tokenizer):
    text = output.text
    truncated = output.finish_reason == "length"
    closed = THINK_CLOSE in text
    think, _, answer = text.partition(THINK_CLOSE)
    think_tokens = len(tokenizer.encode(think, add_special_tokens=False)) if closed else None
    correct = closed and is_correct(prompt, answer)
    return {
        "id": prompt["id"],
        "bench": prompt["bench"],
        "sample": sample,
        "tokens": len(output.token_ids),
        "think_tokens": think_tokens,
        "closed": closed,
        "truncated": truncated,
        "correct": correct,
        "strict": bool(correct and not truncated and think_tokens >= MIN_THINK_TOKENS),
        "text": text,
    }


def percent(rows, key):
    return 100 * sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def summarize(rows):
    finished = [row for row in rows if row["closed"] and not row["truncated"]]
    return {
        "n": len(rows),
        "acc": percent(rows, "correct"),
        "strict": percent(rows, "strict"),
        "trunc": percent(rows, "truncated"),
        "closed": percent(rows, "closed"),
        "acc_finished": percent(finished, "correct"),
        "mean_tokens": sum(row["tokens"] for row in rows) / max(len(rows), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    args = parser.parse_args()

    os.makedirs(args.out)
    with open(args.prompts) as handle:
        prompts = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=34816,
        max_num_seqs=args.max_seqs,
        gpu_memory_utilization=args.gpu_util,
        language_model_only=True,
    )

    meta, texts, params = [], [], []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for sample in range(args.samples):
            meta.append((prompt, sample))
            texts.append(rendered)
            params.append(
                SamplingParams(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    presence_penalty=1.5,
                    max_tokens=args.max_tokens,
                    seed=args.seed + 7919 * prompt["id"] + sample,
                )
            )

    started = time.time()
    outputs = llm.generate(texts, params)
    seconds = time.time() - started

    rows = [score_row(prompt, sample, out.outputs[0], tokenizer) for (prompt, sample), out in zip(meta, outputs)]
    with open(os.path.join(args.out, "rows.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {"model": args.model, "seed": args.seed, "samples": args.samples, "seconds": seconds, "all": summarize(rows)}
    for bench in sorted({row["bench"] for row in rows}):
        summary[bench] = summarize([row for row in rows if row["bench"] == bench])
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
