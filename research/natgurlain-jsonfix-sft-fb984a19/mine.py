"""Sample a checkpoint on training-pool prompts at the evaluation sampler and record per-prompt outcomes.

Two uses, same code:
  * probe  - a fixed held-out prompt set, run on every candidate, compared pairwise by prompt;
  * mining - a larger disjoint set, run on the parent, to find prompts where it truncates or
             splits its answers (the only prompts that carry a GRPO learning signal).

Prompts come from the public training splits used by grpo.py. No leaderboard or hidden-test data.
Output is JSONL, one line per prompt, appended chunk by chunk so an interrupted run resumes.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import grpo


def key_of(item):
    return hashlib.sha1(item["prompt"].encode()).hexdigest()[:16]


def build(mix, n_prompts, seed, exclude):
    skip = grpo.excluded_questions()
    builders = {
        "math": lambda: [x for x in grpo.math_pool() if x["prompt"][:-len(grpo.SUFFIX)] not in skip],
        "choice": lambda: grpo.choice_pool(broad=True),
        "ifeval": grpo.ifeval_pool,
        "logic": grpo.logic_pool,
    }
    rng = random.Random(seed)
    chosen = []
    for name, share in mix.items():
        pool = [x for x in builders[name]() if key_of(x) not in exclude]
        want = round(n_prompts * share)
        picked = rng.sample(pool, min(want, len(pool)))
        for it in picked:
            chosen.append(dict(it, pool=name, key=key_of(it)))
        print(f"pool {name}: {len(pool)} available, {len(picked)} chosen", flush=True)
    rng.shuffle(chosen)
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--prompts", type=int, default=1000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sample-seed", type=int, default=20261104)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--mix", default="math=0.35,choice=0.30,ifeval=0.25,logic=0.10")
    p.add_argument("--exclude", action="append", default=[], help="JSONL files whose prompt keys are left out")
    p.add_argument("--items", default="", help="reuse the exact prompt list of an earlier output file")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-text", action="store_true", help="keep the generated text (small diagnostic sets only)")
    p.add_argument("--temperature", type=float, default=1.0, help="test sampler uses 1.0")
    p.add_argument("--presence", type=float, default=1.5, help="test sampler uses 1.5")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.open() if line.strip()}
    if a.items:
        items = [json.loads(line)["item"] for line in open(a.items) if line.strip()]
    else:
        exclude = set()
        for f in a.exclude:
            exclude |= {json.loads(line)["key"] for line in open(f) if line.strip()}
        mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
        items = build(mix, a.prompts, a.seed, exclude)
    todo = [it for it in items if it["key"] not in done]
    print(f"{len(items)} prompts, {len(done)} already done, {len(todo)} to run, k={a.k}", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_tokens + 2048, seed=0,
              gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.80")), max_num_seqs=256)
    scorer = ProcessPoolExecutor(16, mp_context=multiprocessing.get_context("spawn"))
    t_start = time.time()
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for it in chunk]
        # the seed depends on the prompt set position only, so every model sees the same request stream
        sampling = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95, top_k=20, presence_penalty=a.presence,
                                  max_tokens=a.max_tokens, skip_special_tokens=False,
                                  seed=a.sample_seed + items.index(chunk[0]))
        gens = llm.generate(prompts, sampling, use_tqdm=False)
        jobs = []
        for it, g in zip(chunk, gens):
            for c in g.outputs:
                jobs.append((it, c.text, len(c.token_ids) >= a.max_tokens))
        rewards = list(scorer.map(grpo.score_one, jobs, chunksize=16))
        with out.open("a") as f:
            for i, (it, g) in enumerate(zip(chunk, gens)):
                f.write(json.dumps({
                    "key": it["key"], "pool": it["pool"], "src": it["src"],
                    "reward": rewards[i * a.k:(i + 1) * a.k],
                    "tokens": [len(c.token_ids) for c in g.outputs],
                    "closed": [("</think>" in c.text) for c in g.outputs],
                    "item": it,
                    **({"text": [c.text for c in g.outputs]} if a.save_text else {}),
                }) + "\n")
        n = c0 + len(chunk)
        el = time.time() - t_start
        print(json.dumps({"done": n, "of": len(todo), "elapsed_s": round(el), "eta_s": round(el / n * (len(todo) - n)),
                          "chunk_reward": round(sum(rewards) / len(rewards), 4),
                          "chunk_trunc": round(sum(j[2] for j in jobs) / len(jobs), 4)}), flush=True)


if __name__ == "__main__":
    main()
