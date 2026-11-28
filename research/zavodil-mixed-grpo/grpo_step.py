"""One GRPO update at the evaluation sampler, from traces already generated with run_proxy.

usage: grpo_step.py MASTER.safetensors ROWS.jsonl PROMPTS.json OUT.safetensors
         [--lr 2e-5] [--layers 12-23] [--positions 192] [--max-tokens 8192] [--kl 0.0]

Each prompt is a group of samples from the current policy. Reward is 1 for a correct
generation (closed, right answer) and 0 otherwise; the advantage is the reward minus the group mean, so prompts where
all samples agree contribute nothing and are skipped. The gradient step raises the log
probability of tokens in above-average samples and lowers it in below-average ones.

The log probability uses the sampler's presence penalty on tokens already generated, as the
evaluation does, with top-k and top-p ignored. MASTER keeps float32 weights across rounds, so
updates smaller than BF16 rounding still accumulate; run_proxy loads a BF16 export of it.
"""
import argparse
import json
import random
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

PRESENCE_PENALTY = 1.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master")
    parser.add_argument("rows")
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--layers", default="12-23")
    parser.add_argument("--positions", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200000)
    parser.add_argument("--ulp-steps", type=float, default=1.0)
    parser.add_argument("--reward", default="correct", help="row field used as the reward")
    parser.add_argument("--step-norm", type=float, default=0.0, help="dense step of this total L2 norm, per-tensor RMS normalized; replaces the sparse step")
    parser.add_argument("--skip-first", action="store_true", help="leave the first generated token out of the loss")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.config_dir)
    prompts = {row["id"]: row.get("messages") or [{"role": "user", "content": row["prompt"]}] for row in json.load(open(args.prompts))}
    groups = {}
    for line in open(args.rows):
        row = json.loads(line)
        groups.setdefault(row["id"], []).append(row)

    items = []
    for rows in groups.values():
        mean = sum(r[args.reward] for r in rows) / len(rows)
        if 0 < mean < 1:
            items += [(r, float(r[args.reward]) - mean) for r in rows]
    scale = (sum(a * a for _, a in items) / max(len(items), 1)) ** 0.5 or 1.0
    random.shuffle(items)
    if args.limit:
        items = items[: args.limit]
    all_rows = [r for rows in groups.values() for r in rows]
    report = {
        "groups": len(groups), "groups_with_signal": sum(0 < sum(r[args.reward] for r in v) / len(v) < 1 for v in groups.values()),
        "reward": round(100 * sum(r[args.reward] for r in all_rows) / len(all_rows), 2),
        "strict": round(100 * sum(r["strict"] for r in all_rows) / len(all_rows), 2),
        "trunc": round(100 * sum(r["truncated"] for r in all_rows) / len(all_rows), 2),
        "traces_used": len(items),
    }
    print(json.dumps(report), flush=True)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.config_dir, torch_dtype=torch.float32)
    state = load_file(args.master)
    missing, _ = model.load_state_dict(state, strict=False)
    del state
    if any("language_model" in key and "embed" not in key for key in missing):
        raise SystemExit(f"master is missing model tensors: {missing[:5]}")
    model = model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lo, hi = (int(x) for x in args.layers.split("-"))
    trainable = []
    for name, param in model.named_parameters():
        use = any(name.startswith(f"model.language_model.layers.{i}.") for i in range(lo, hi + 1))
        param.requires_grad_(use)
        if use:
            trainable.append(param)
    print(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)
    model.train()

    started = time.time()
    total_tokens = max(sum(row["tokens"] for row, _ in items), 1)
    for index, (row, advantage) in enumerate(items):
        rendered = tokenizer.apply_chat_template(prompts[row["id"]], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        generated = tokenizer.encode(row["text"], add_special_tokens=False)[: args.max_tokens]
        input_ids = torch.tensor([prompt_ids + generated], device="cuda")
        count = len(generated)
        start = 1 if args.skip_first else 0
        if count <= start:
            continue
        chosen = sorted(random.sample(range(start, count), min(args.positions, count - start)))
        positions = torch.tensor([len(prompt_ids) + i - 1 for i in chosen], device="cuda")
        targets = torch.tensor([generated[i] for i in chosen], device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids).last_hidden_state[0, positions]
            logits = model.lm_head(hidden).float()

        penalty = torch.zeros_like(logits, dtype=torch.bool)
        seen = torch.zeros(logits.shape[1], dtype=torch.bool, device="cuda")
        cursor = 0
        gen_tensor = torch.tensor(generated, device="cuda")
        for slot, i in enumerate(chosen):
            seen[gen_tensor[cursor:i]] = True
            cursor = i
            penalty[slot] = seen
        logp = torch.log_softmax(logits - PRESENCE_PENALTY * penalty.float(), -1).gather(1, targets[:, None])[:, 0]
        # Sampled positions estimate the sum of log probabilities over the whole trace, so long
        # traces carry their full weight: a truncated run counts for all of its tokens.
        loss = -(advantage / scale) * logp.sum() * ((count - start) / len(chosen)) / total_tokens
        loss.backward()
        if index % 25 == 0:
            print(f"trace {index}/{len(items)} adv {advantage:+.2f} tokens {count} {time.time() - started:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)

    if args.step_norm:
        # Dense step: each tensor moves by its gradient over the gradient RMS, clipped at 3, then
        # the whole step is scaled to a fixed L2 norm. Round-to-round progress accumulates in the
        # float32 master, and the BF16 export keeps what crosses a rounding boundary.
        steps = []
        with torch.no_grad():
            for p in trainable:
                if p.grad is not None:
                    steps.append((p, (p.grad / (p.grad.pow(2).mean().sqrt() + 1e-20)).clamp(-3, 3)))
            total = sum(float(s.pow(2).sum()) for _, s in steps) ** 0.5
            for p, s in steps:
                p.add_(-args.step_norm * s / total)
        print(f"moved dense step norm {args.step_norm}, elapsed {time.time() - started:.0f}s", flush=True)
    else:
        # Sparse update: only the entries with the largest gradient relative to their BF16 rounding
        # step move, by at least that step. Submitted weights are BF16, so a dense small update would
        # survive only as rounding noise whose norm exceeds the update itself (see TRIED.md).
        ratios, sizes = [], []
        with torch.no_grad():
            for p in trainable:
                if p.grad is None:
                    continue
                ulp = torch.maximum(p.abs(), torch.full_like(p, 2.0 ** -14)) * 2.0 ** -7
                ratios.append((p.grad.abs() / ulp).flatten().float().cpu())
            joined = torch.cat(ratios)
            threshold = torch.topk(joined, min(args.top_k, joined.numel())).values[-1]
            del joined, ratios
            moved = 0
            for p in trainable:
                if p.grad is None:
                    continue
                ulp = torch.maximum(p.abs(), torch.full_like(p, 2.0 ** -14)) * 2.0 ** -7
                mask = (p.grad.abs() / ulp) >= threshold.to(p.device)
                p.add_(-args.ulp_steps * ulp * torch.sign(p.grad) * mask)
                moved += int(mask.sum())
        print(f"moved {moved} entries by {args.ulp_steps} ulp, elapsed {time.time() - started:.0f}s", flush=True)

    with safe_open(args.master, "pt") as handle:
        names = list(handle.keys())
        out = {name: handle.get_tensor(name) for name in names}
    fresh = dict(model.state_dict())
    changed = 0
    for name in names:
        if name in fresh and fresh[name].shape == out[name].shape:
            value = fresh[name].detach().cpu().to(torch.float32)
            if not torch.equal(value, out[name].float()):
                out[name] = value.contiguous()
                changed += 1
    save_file(out, args.out)
    print(f"saved {args.out}, changed {changed} tensors", flush=True)


if __name__ == "__main__":
    main()
