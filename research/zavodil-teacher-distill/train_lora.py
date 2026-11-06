"""Distill teacher traces into the student with a LoRA, anchored to the student by KL.

usage: train_lora.py STUDENT_DIR TRACES.jsonl PROMPTS.json OUT_DIR
         [--rank 16] [--lr 1e-4] [--epochs 1] [--max-len 6144] [--kl 0.1] [--layers 12-23]

Selection: correct, closed, not truncated, thinking between 200 and 4000 tokens. One
trace per prompt (the shortest that qualifies). Loss: cross-entropy on the generated part,
plus kl * KL(student_base || student_lora) on the same tokens so the model keeps its
behaviour where the teacher trace carries no signal. The merged checkpoint is written
to OUT_DIR/model with the student's tokenizer files, ready for run_proxy and delta patch.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

THINK_CLOSE = "</think>"


def select(traces, tokenizer, min_think, max_think):
    best = {}
    for row in traces:
        if not (row["correct"] and row["closed"] and not row["truncated"]):
            continue
        think = row["think_tokens"] or 0
        if not (min_think <= think <= max_think):
            continue
        if row["id"] not in best or row["tokens"] < best[row["id"]]["tokens"]:
            best[row["id"]] = row
    return list(best.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("student")
    parser.add_argument("traces")
    parser.add_argument("prompts")
    parser.add_argument("out")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=6144)
    parser.add_argument("--kl", type=float, default=0.1)
    parser.add_argument("--layers", default="12-23")
    parser.add_argument("--min-think", type=int, default=200)
    parser.add_argument("--max-think", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    prompts = {r["id"]: r["prompt"] for r in json.load(open(args.prompts))}
    traces = [json.loads(line) for line in open(args.traces)]
    chosen = select(traces, tokenizer, args.min_think, args.max_think)
    random.shuffle(chosen)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"selected {len(chosen)} traces from {len(traces)}", flush=True)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.student, torch_dtype=dtype).to(args.device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lo, hi = (int(x) for x in args.layers.split("-"))
    config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
        target_modules=[name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                        and any(name.startswith(f"model.language_model.layers.{i}.") for i in range(lo, hi + 1))
                        and (".mlp." in name or ".self_attn." in name or ".linear_attn." in name)],
    )
    print(f"lora targets {len(config.target_modules)} linear modules", flush=True)
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable / 1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(chosen)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))

    step = 0
    for epoch in range(args.epochs):
        for row in chosen:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompts[row["id"]]}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=True)
            p_ids = tokenizer.encode(rendered, add_special_tokens=False)
            g_ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
            ids = (p_ids + g_ids)[: args.max_len]
            input_ids = torch.tensor([ids]).to(args.device)
            labels = input_ids.clone()
            labels[0, : len(p_ids)] = -100

            with torch.no_grad():
                with model.disable_adapter():
                    base_logits = model(input_ids=input_ids).logits[0, len(p_ids) - 1 : -1].float()
            logits = model(input_ids=input_ids).logits[0]
            ce = torch.nn.functional.cross_entropy(logits[:-1].float(), labels[0, 1:], ignore_index=-100)
            student = torch.log_softmax(logits[len(p_ids) - 1 : -1].float(), -1)
            base = torch.log_softmax(base_logits, -1)
            kl = (base.exp() * (base - student)).sum(-1).mean()
            loss = ce + args.kl * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == steps:
                print(f"step {step}/{steps} ce {ce.item():.3f} kl {kl.item():.4f} tokens {len(ids)}", flush=True)

    merged = model.merge_and_unload()
    export_checkpoint(merged, Path(args.student), out / "model")
    print("saved", out / "model")


def export_checkpoint(merged, student_dir, out_dir):
    """Write the merged weights into the student's exact tensor layout: same names, dtypes
    and untouched tensors (including the MTP head), so patch building sees only real changes."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = {k: v.detach().cpu() for k, v in merged.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, changed = {}, 0
    with safe_open(student_dir / "model.safetensors", "pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            base = handle.get_tensor(key)
            new = state.get(key)
            if new is not None and new.shape == base.shape and not torch.equal(new.to(base.dtype), base):
                tensors[key] = new.to(base.dtype).contiguous()
                changed += 1
            else:
                tensors[key] = base
    save_file(tensors, out_dir / "model.safetensors", metadata=metadata or {"format": "pt"})
    for item in student_dir.iterdir():
        if item.name not in ("model.safetensors", "delta-model.json"):
            shutil.copy(item, out_dir / item.name)
    print(f"exported {changed} changed tensors of {len(tensors)}", flush=True)


if __name__ == "__main__":
    main()
