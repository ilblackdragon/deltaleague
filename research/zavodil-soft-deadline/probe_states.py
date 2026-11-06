"""Teacher-force existing generations and record hidden states for edit design.

usage: probe_states.py MODEL_DIR ROWS.jsonl OUT.npz --per-group 8 --device cpu

Groups: strict (correct, closed), overflow (closed, truncated), never (not closed).
Records, subsampled every STRIDE positions: the final-normed hidden state, the input
of layer 23's down_proj, the position, the phase (0 think, 1 answer) and the group.
At every position: logits of </think>, <|im_end|> and the first token of "\\boxed".
"""
import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

STRIDE = 4
THINK_CLOSE = 248069
LAYER = 23


def pick(rows, per_group, rng):
    groups = {
        "strict": [r for r in rows if r["strict"]],
        "overflow": [r for r in rows if r["closed"] and r["truncated"]],
        "never": [r for r in rows if not r["closed"]],
    }
    chosen = []
    for name, items in groups.items():
        for row in rng.sample(items, min(per_group, len(items))):
            chosen.append((name, row))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("rows")
    parser.add_argument("out")
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompts", required=True)
    args = parser.parse_args()

    torch.set_num_threads(14)
    rng = random.Random(args.seed)
    prompts = {r["id"]: r["prompt"] for r in json.load(open(args.prompts))}
    rows = [json.loads(line) for line in open(args.rows)]
    for row in rows:
        row["prompt"] = prompts[row["id"]]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    language = model.model.language_model
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    boxed_id = tokenizer.encode("\\boxed{", add_special_tokens=False)[0]
    watch = [THINK_CLOSE, end_id, boxed_id]
    print("watch tokens", [tokenizer.decode([t]) for t in watch], flush=True)

    captured = {}
    layer = language.layers[LAYER]
    layer.mlp.down_proj.register_forward_hook(lambda module, inputs, output: captured.__setitem__("a", inputs[0].detach()))

    store = {k: [] for k in ("h", "a", "pos", "phase", "group", "trace", "logit_pos", "logits", "logit_trace", "logit_phase")}
    chosen = pick(rows, args.per_group, rng)
    for index, (group, row) in enumerate(chosen):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        gen_ids = tokenizer.encode(row["text"], add_special_tokens=False)
        ids = torch.tensor([prompt_ids + gen_ids], device=args.device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-1][0]
        logits = out.logits[0]
        a = captured["a"][0]
        start = len(prompt_ids)
        close = [i for i, t in enumerate(gen_ids) if t == THINK_CLOSE]
        close_pos = start + close[0] if close else None
        total = ids.shape[1]
        for p in range(start, total, STRIDE):
            store["h"].append(h[p].to(torch.float16).numpy())
            store["a"].append(a[p].to(torch.float16).numpy())
            store["pos"].append(p - start)
            store["phase"].append(0 if close_pos is None or p < close_pos else 1)
            store["group"].append(group)
            store["trace"].append(index)
        sel = logits[start:total][:, watch].float().numpy()
        store["logits"].append(sel)
        store["logit_pos"].append(np.arange(total - start))
        store["logit_trace"].append(np.full(total - start, index))
        store["logit_phase"].append(np.array([0 if close_pos is None or p < close_pos else 1 for p in range(start, total)]))
        print(f"{index} {group} tokens={total - start} close_at={None if close_pos is None else close_pos - start}", flush=True)

    np.savez_compressed(
        args.out,
        h=np.stack(store["h"]), a=np.stack(store["a"]), pos=np.array(store["pos"]), phase=np.array(store["phase"]),
        group=np.array(store["group"]), trace=np.array(store["trace"]),
        logits=np.concatenate(store["logits"]), logit_pos=np.concatenate(store["logit_pos"]),
        logit_trace=np.concatenate(store["logit_trace"]), logit_phase=np.concatenate(store["logit_phase"]),
        watch=np.array(watch),
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
