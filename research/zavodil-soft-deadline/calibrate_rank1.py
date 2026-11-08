"""Measure how a rank-1 edit moves target logits, by phase, on real traces.

usage: calibrate_rank1.py BASE_DIR EDITED_DIR ROWS.jsonl PROMPTS.json [--n 3] [--max-tokens 2500]

Runs both checkpoints on the same traces and reports the mean logit change of
</think>, <|im_end|> and "\\boxed" in the thinking phase and in the answer phase,
plus the change in log-probability of the actually generated tokens.
"""
import argparse
import json
import random

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

THINK_CLOSE = 248069


def forward(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits[0].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("edited")
    parser.add_argument("rows")
    parser.add_argument("prompts")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2500)
    args = parser.parse_args()
    torch.set_num_threads(14)

    prompts = {r["id"]: r["prompt"] for r in json.load(open(args.prompts))}
    rows = [json.loads(l) for l in open(args.rows)]
    rng = random.Random(3)
    picked = rng.sample([r for r in rows if r["strict"]], args.n) + rng.sample([r for r in rows if r["closed"] and r["truncated"]], args.n)

    tok = AutoTokenizer.from_pretrained(args.base)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    boxed_id = next(t for t in tok.encode("\\boxed{", add_special_tokens=False) if "boxed" in tok.decode([t]))
    watch = {"close": THINK_CLOSE, "end": end_id, "boxed": boxed_id}
    models = {name: Qwen3_5ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32).eval() for name, path in (("base", args.base), ("edited", args.edited))}

    sums = {phase: {k: [] for k in list(watch) + ["logp_generated"]} for phase in ("think", "answer_strict", "answer_overflow")}
    for row in picked:
        kind = "answer_strict" if row["strict"] else "answer_overflow"
        rendered = tok.apply_chat_template([{"role": "user", "content": prompts[row["id"]]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p_ids = tok.encode(rendered, add_special_tokens=False)
        g_ids = tok.encode(row["text"], add_special_tokens=False)[: args.max_tokens]
        ids = torch.tensor([p_ids + g_ids])
        logits = {name: forward(m, ids)[len(p_ids) - 1 : -1] for name, m in models.items()}
        close = [i for i, t in enumerate(g_ids) if t == THINK_CLOSE]
        split = close[0] if close else len(g_ids)
        target = torch.tensor(g_ids)
        for phase, sl in (("think", slice(0, split)), (kind, slice(split, len(g_ids)))):
            if sl.start >= len(g_ids):
                continue
            for name, tid in watch.items():
                sums[phase][name].append(float((logits["edited"][sl, tid] - logits["base"][sl, tid]).mean()))
            lp = {n: torch.log_softmax(l[sl], -1).gather(1, target[sl, None]).mean() for n, l in logits.items()}
            sums[phase]["logp_generated"].append(float(lp["edited"] - lp["base"]))
    out = {phase: {k: round(sum(v) / len(v), 3) for k, v in d.items() if v} for phase, d in sums.items()}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
