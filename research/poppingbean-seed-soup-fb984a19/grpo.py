import argparse
import gc
import json
import multiprocessing
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CHOICE_SUFFIX = "\nPlease reason step by step, and put the letter of the correct option within \\boxed{}."
LETTERS = "ABCDE"
LOGIC_TASKS = ("strategyqa", "logic_grid_puzzle", "analytic_entailment", "logical_args", "conceptual_combinations",
               "physical_intuition", "epistemic_reasoning", "presuppositions_as_nli", "implicatures", "elementary_math_qa")


def math_pool():
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    math = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    from probe import boxed
    out = [{"prompt": d["question"] + SUFFIX, "scorer": "math", "answer": d["answer"].split("####")[-1].strip(), "src": "gsm8k"} for d in gsm]
    out += [{"prompt": d["problem"] + SUFFIX, "scorer": "math", "answer": boxed(d["solution"]), "src": "math"} for d in math]
    return out


def choice_item(question, labels, texts, key, src):
    if key not in labels or len(labels) > 5:
        return None
    options = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
    return {"prompt": f"{question}\n\n{options}{CHOICE_SUFFIX}", "scorer": "choice", "answer": LETTERS[labels.index(key)], "src": src}


def choice_pool(broad=False):
    out = []
    for d in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"):
        out.append(choice_item(d["question"], d["choices"]["label"], d["choices"]["text"], d["answerKey"], "arc"))
    for d in load_dataset("tau/commonsense_qa", split="train"):
        out.append(choice_item(d["question"], d["choices"]["label"], d["choices"]["text"], d["answerKey"], "csqa"))
    for d in load_dataset("allenai/openbookqa", "main", split="train"):
        out.append(choice_item(d["question_stem"], d["choices"]["label"], d["choices"]["text"], d["answerKey"], "obqa"))
    if broad:
        for d in load_dataset("allenai/winogrande", "winogrande_xl", split="train"):
            out.append(choice_item(d["sentence"] + "\nWhich option correctly fills the blank (_)?", ["1", "2"], [d["option1"], d["option2"]], d["answer"], "wino"))
        for d in load_dataset("Rowan/hellaswag", split="train"):
            out.append(choice_item(d["ctx"] + "\nWhich ending is most likely?", ["0", "1", "2", "3"], d["endings"], str(d["label"]), "hswag"))
        for d in load_dataset("google/boolq", split="train"):
            out.append(choice_item(f"{d['passage']}\n\nQuestion: {d['question']}?", ["yes", "no"], ["Yes", "No"], "yes" if d["answer"] else "no", "boolq"))
        for d in load_dataset("cais/mmlu", "auxiliary_train", split="train"):
            d = d.get("train", d)
            out.append(choice_item(d["question"], ["0", "1", "2", "3"], d["choices"], str(d["answer"]), "mmlu_aux"))
    return [x for x in out if x]


def ifeval_pool():
    held_out = {d["prompt"].strip() for d in load_dataset("google/IFEval", split="train")}
    out = []
    for d in load_dataset("argilla/ifeval-like-data", "filtered", split="train"):
        if d["prompt"].strip() in held_out:
            continue
        kwargs = json.loads(d["kwargs"]) if isinstance(d["kwargs"], str) else d["kwargs"]
        out.append({"prompt": d["prompt"], "scorer": "ifeval", "answer": None, "instruction_id_list": d["instruction_id_list"],
                    "kwargs": kwargs, "src": "ifeval"})
    return out


def logic_pool():
    out = []
    for task in LOGIC_TASKS:
        try:
            rows = load_dataset("tasksource/bigbench", task, split="train")
        except Exception as e:
            print(f"skip logic task {task}: {e!r}", flush=True)
            continue
        kept = 0
        for d in rows:
            targets, scores = d.get("multiple_choice_targets") or [], d.get("multiple_choice_scores") or []
            if 2 <= len(targets) <= 5 and sum(scores) == 1 and kept < 3000:
                lines = [ln for ln in d["inputs"].strip().split("\n") if not ln.strip().startswith("choice:")]
                while lines and lines[-1].strip() in ("A:", "Answer:", ""):
                    lines.pop()
                labels = [str(i) for i in range(len(targets))]
                out.append(choice_item("\n".join(lines), labels, targets, str(scores.index(1)), f"logic_{task}"))
                kept += 1
    return [x for x in out if x]


def excluded_questions():
    from probe import problems
    seen = set()
    for n, seed in ((256, 1), (256, 3), (512, 3), (256, 7)):
        seen.update(it["question"] for it in problems(n, seed))
    return seen


def score_one(args):
    item, text, truncated = args
    from delta_cli.scoring import score_answer
    if truncated or "</think>" not in text:
        return 0.0
    try:
        return score_answer(item, text)
    except Exception:
        return 0.0


def reload_weights(self, path):
    from safetensors.torch import load_file
    self.model_runner.model.load_weights(weights=load_file(path).items())
    return True


def sampler_logprob(logits, t, pos, first_seen, presence=1.5, top_k=20, top_p=0.95, pool=64):
    cand = logits.detach().topk(pool, dim=-1).indices
    pen = logits.gather(-1, cand) - presence * (first_seen[cand] < pos[:, None]).to(logits.dtype)
    top, order = pen.topk(top_k, dim=-1)
    probs = top.detach().softmax(-1)
    keep = (probs.cumsum(-1) - probs) < top_p
    lse = torch.logsumexp(top.masked_fill(~keep, float("-inf")), -1)
    chosen = logits.gather(-1, t[:, None])[:, 0] - presence * (first_seen[t] < pos).to(logits.dtype)
    kept_ids = cand.gather(-1, order).masked_fill(~keep, -1)
    valid = (kept_ids == t[:, None]).any(-1)
    return chosen - lse, valid


def token_logprobs(model, embed, ids, n_prompt, match_sampler=False):
    hidden = model.model.language_model(input_ids=ids).last_hidden_state[0, n_prompt - 1:-1]
    targets = ids[0, n_prompt:]
    n = targets.shape[0]
    first_seen = torch.full((embed.shape[0],), n, dtype=torch.long, device=targets.device)
    first_seen.scatter_reduce_(0, targets, torch.arange(n, device=targets.device), reduce="amin")
    lps, ents, valids = [], [], []
    for s in range(0, n, 1024):
        def chunk(h, t, pos):
            logits = (h @ embed.T).float()
            logp = logits.log_softmax(-1)
            entropy = -(logp.detach().exp() * logp.detach()).sum(-1)
            if match_sampler:
                lp, valid = sampler_logprob(logits, t, pos, first_seen)
                return lp, entropy, valid
            return logp.gather(-1, t[:, None])[:, 0], entropy, torch.ones_like(t, dtype=torch.bool)
        pos = torch.arange(s, min(s + 1024, n), device=targets.device)
        lp, ent, valid = torch.utils.checkpoint.checkpoint(chunk, hidden[s:s + 1024], targets[s:s + 1024], pos, use_reentrant=False)
        lps.append(lp)
        ents.append(ent)
        valids.append(valid)
    return torch.cat(lps), torch.cat(ents), torch.cat(valids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("init")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--choice-share", type=float, default=0.35)
    p.add_argument("--save-at", default="10,20")
    p.add_argument("--norm-len", type=int, default=4096)
    p.add_argument("--top-entropy", type=float, default=0.0)
    p.add_argument("--train-max-tokens", type=int, default=8192)
    p.add_argument("--match-sampler", action="store_true")
    p.add_argument("--stop-on-decline", type=float, default=0.0)
    p.add_argument("--len-shape", type=int, default=0, help="correct answers longer than this lose up to half their reward at the cap")
    p.add_argument("--seed", type=int, default=0, help="0 reproduces #58/#74; other values change prompt order and sampling seeds")
    p.add_argument("--mix", default="", help="e.g. math=0.35,choice=0.3,ifeval=0.25,logic=0.1")
    a = p.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(a.out)
    live = out / "live"
    live.mkdir(parents=True, exist_ok=True)
    for f in Path(a.init).iterdir():
        if f.is_file() and f.name != "delta-model.json":
            shutil.copy(f, live / f.name)
    with safe_open(str(Path(a.init) / "model.safetensors"), "pt") as f:
        dtypes = {k: f.get_slice(k).get_dtype() for k in f.keys()}
    dtypes = {k: {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[v] for k, v in dtypes.items()}
    scorer = ProcessPoolExecutor(16, mp_context=multiprocessing.get_context("spawn"))
    log = open(out / "train_log.jsonl", "a")
    skip = excluded_questions()
    pool_m = [x for x in math_pool() if x["prompt"][:-len(SUFFIX)] not in skip]
    if a.mix:
        shares = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
        builders = {"math": lambda: pool_m, "choice": lambda: choice_pool(broad=True), "ifeval": ifeval_pool, "logic": logic_pool}
        pools = {k: builders[k]() for k in shares}
    else:
        shares = {"math": 1 - a.choice_share, "choice": a.choice_share}
        pools = {"math": pool_m, "choice": choice_pool()}
    pools = {k: v for k, v in pools.items() if v}
    counts = {k: round(a.prompts * shares[k]) for k in pools}
    rng = random.Random(1234 + a.seed)
    print(f"pools {({k: len(v) for k, v in pools.items()})} per step {counts} excluded={len(skip)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.init)
    model = AutoModelForImageTextToText.from_pretrained(a.init, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    embed = model.model.language_model.embed_tokens.weight
    held = set(model.state_dict())
    with safe_open(str(Path(a.init) / "model.safetensors"), "pt") as f:
        frozen = {k: f.get_tensor(k) for k in dtypes if k not in held}
    print(f"untrained checkpoint tensors kept as-is: {len(frozen)}", flush=True)

    def make_llm():
        return LLM(str(live), dtype="bfloat16", max_model_len=8192 + 2048, seed=0,
                   gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.35")), max_num_seqs=256)

    llm = make_llm()
    sampling = SamplingParams(n=a.group, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                              max_tokens=a.train_max_tokens, skip_special_tokens=False)
    save_at = {int(s) for s in a.save_at.split(",") if s}
    sync_path = str(live / "model.safetensors")

    history = []
    for step in range(1, a.steps + 1):
        t0 = time.time()
        batch = [it for k in pools for it in rng.sample(pools[k], counts[k])]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for it in batch]
        sampling.seed = 1000 + step + 100000 * a.seed
        gens = llm.generate(prompts, sampling, use_tqdm=False)
        t_gen = time.time() - t0
        jobs, seqs = [], []
        for it, prompt, g in zip(batch, prompts, gens):
            p_ids = list(g.prompt_token_ids)
            for c in g.outputs:
                truncated = len(c.token_ids) >= a.train_max_tokens
                jobs.append((it, c.text, truncated))
                seqs.append((it, p_ids, list(c.token_ids), truncated))
        rewards = list(scorer.map(score_one, jobs, chunksize=16))
        if a.len_shape:
            span = max(1, a.train_max_tokens - a.len_shape)
            rewards = [r * (1 - 0.5 * min(1.0, max(0, len(s[2]) - a.len_shape) / span)) for r, s in zip(rewards, seqs)]
        stats = {"step": step, "reward": sum(rewards) / len(rewards),
                 "trunc": sum(s[3] for s in seqs) / len(seqs),
                 "tokens": sum(len(s[2]) for s in seqs) / len(seqs)}
        for src in sorted({s[0]["src"].split("_")[0] if s[0]["src"].startswith("logic") else s[0]["src"] for s in seqs}):
            r = [x for x, s in zip(rewards, seqs) if s[0]["src"] == src or s[0]["src"].startswith(src + "_")]
            if r:
                stats[f"r_{src}"] = round(sum(r) / len(r), 3)
        train = []
        for gi in range(len(batch)):
            grp = rewards[gi * a.group:(gi + 1) * a.group]
            mean = sum(grp) / len(grp)
            for j, r in enumerate(grp):
                if r != mean:
                    train.append((r - mean, seqs[gi * a.group + j]))
        opt.zero_grad(set_to_none=True)
        n_valid = n_tok = 0
        scale = 1.0 / (len(seqs) * a.norm_len)
        for adv, (_, p_ids, g_ids, _) in train:
            ids = torch.tensor([p_ids + g_ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, ent, valid = token_logprobs(model, embed, ids, len(p_ids), a.match_sampler)
            lp = torch.where(valid, lp, torch.zeros_like(lp))
            n_valid += int(valid.sum())
            n_tok += valid.numel()
            if a.top_entropy > 0:
                k = max(1, int(a.top_entropy * ent.numel()))
                lp = lp * (ent >= ent.topk(k).values[-1]).to(lp.dtype)
            (-(adv * scale) * lp.sum()).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item() if train else 0.0
        if train:
            opt.step()
        current = model.state_dict()
        state = {k: frozen[k] if k in frozen else current[k].detach().to(dt).contiguous() for k, dt in dtypes.items()}
        save_file(state, sync_path, metadata={"format": "pt"})
        try:
            llm.collective_rpc(reload_weights, args=(sync_path,))
            llm.reset_prefix_cache()
        except Exception as e:
            print(f"in-place reload failed ({e!r}); restarting the engine", flush=True)
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            llm = make_llm()
        if step in save_at:
            ck = out / f"step{step}"
            ck.mkdir(exist_ok=True)
            for f in Path(a.init).iterdir():
                if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"):
                    shutil.copy(f, ck / f.name)
            shutil.copy(sync_path, ck / "model.safetensors")
        del state
        gc.collect()
        stats.update({"train_seqs": len(train), "grad_norm": round(gnorm, 4), "gen_s": round(t_gen), "step_s": round(time.time() - t0),
                      "sampler_coverage": round(n_valid / n_tok, 4) if n_tok else None})
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        history.append(stats["reward"])
        if a.stop_on_decline and len(history) >= 8 and sum(history[-4:]) / 4 < sum(history[:4]) / 4 - a.stop_on_decline:
            print(json.dumps({"stopped": step, "reason": "training reward fell", "first4": sum(history[:4]) / 4, "last4": sum(history[-4:]) / 4}), flush=True)
            ck = out / f"step{step}"
            ck.mkdir(exist_ok=True)
            for f in Path(a.init).iterdir():
                if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"):
                    shutil.copy(f, ck / f.name)
            shutil.copy(sync_path, ck / "model.safetensors")
            break


if __name__ == "__main__":
    main()
