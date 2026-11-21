"""Check an edited checkpoint in bfloat16 arithmetic, the dtype the platform computes in.

Loads the parent and the edited BF16 checkpoints one after the other with dtype=bfloat16
(stock transformers, no fast paths), teacher-forces the first N generated tokens of a few
stored completions through each, and reports the effective logit gain of the edited model
(regression slope on the parent's logits over the parent's top 64 tokens) per family and phase.
CPU bfloat16 kernels are not the GPU's, so this checks the edit's tolerance to 8-bit mantissas
in every activation, not bit-exact platform behaviour.

usage: python verify_bf16.py PARENT_DIR EDITED.safetensors JOBS RUN [--per-family 2] [--tokens 400] [--families a,b]
"""
import collections
import json
import os
import sys
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from scorelib import load_rows, outcome

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
for name in ("--per-family", "--tokens", "--families"):
    if name in sys.argv:
        args.remove(sys.argv[sys.argv.index(name) + 1])
parent_dir, edited_file, jobs_path, run_dir = (os.path.expanduser(a) for a in args[:4])
per_family, n_tokens = int(opt("--per-family", 2)), int(opt("--tokens", 400))
want = set(filter(None, opt("--families", "").split(",")))
THINK_CLOSE = 248069

tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)
jobs = {j["id"]: j for j in map(json.loads, open(jobs_path))}
taken, rows = collections.Counter(), []
for r in sorted(load_rows(run_dir), key=lambda r: (r["id"], r["sample"])):
    fam = jobs[r["id"]]["family"]
    close = r["tokens"].index(THINK_CLOSE) if THINK_CLOSE in r["tokens"] else -1
    # keep completions whose answer starts inside the window, so both phases are covered
    if outcome(r) == "ok" and (not want or fam in want) and taken[fam] < per_family and 40 < close < n_tokens - 40:
        taken[fam] += 1
        rows.append((fam, close, r))


def encode(r):
    text = tok.apply_chat_template([{"role": "user", "content": jobs[r["id"]]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = tok(text, add_special_tokens=False).input_ids
    return prompt, r["tokens"][:n_tokens]


@torch.inference_mode()
def logits_of(model_dir, at=None):
    """Logits on 64 tokens per position: the model's own top 64, or the token ids given in `at`.

    Only these are kept: full-vocabulary logits for every position would take 2 GB per completion.
    """
    full = AutoModelForImageTextToText.from_pretrained(model_dir, config=AutoConfig.from_pretrained(model_dir, local_files_only=True),
                                                       local_files_only=True, use_safetensors=True, dtype=torch.bfloat16).eval()
    out = []
    for i, (fam, close, r) in enumerate(rows):
        prompt, gen = encode(r)
        ids = torch.tensor([prompt + gen], dtype=torch.long)
        h = full.model.language_model(input_ids=ids).last_hidden_state[0, len(prompt) - 1:-1]  # states that produced the generated tokens
        vals, idx = [], []
        for a in range(0, h.shape[0], 256):
            z = full.lm_head(h[a:a + 256]).float()
            top = z.topk(64, dim=-1).indices if at is None else at[i][a:a + 256]
            idx.append(top)
            vals.append(z.gather(1, top))
        out.append((torch.cat(idx), torch.cat(vals)))
    del full
    return out


t0 = time.time()
base = logits_of(parent_dir)
print(f"parent in bfloat16: {time.time() - t0:.0f} s for {len(rows)} completions x {n_tokens} tokens", flush=True)
with tempfile.TemporaryDirectory() as tmp:
    for name in os.listdir(parent_dir):
        if name != "model.safetensors":
            os.symlink(os.path.join(parent_dir, name), os.path.join(tmp, name))
    os.symlink(os.path.abspath(edited_file), os.path.join(tmp, "model.safetensors"))
    edit = logits_of(tmp, at=[idx for idx, _ in base])

stats = collections.defaultdict(list)
for (fam, close, r), (_, xb), (_, xe) in zip(rows, base, edit):
    xb, xe = xb - xb.mean(1, keepdim=True), xe - xe.mean(1, keepdim=True)
    slope = (xb * xe).sum(1) / xb.pow(2).sum(1)
    steps = torch.arange(len(slope))
    stats[(fam, "thinking")].append(slope[steps <= close])
    stats[(fam, "answer")].append(slope[steps > close])
print(f"effective gain in bfloat16 arithmetic ({dict(taken)}):")
for key in sorted(stats):
    g = torch.cat(stats[key])
    if len(g):
        q = torch.quantile(g, torch.tensor([0.01, 0.5, 0.99]))
        print(f"   {key[0] + ', ' + key[1]:22s} n={len(g):5d} mean {float(g.mean()):.3f} sd {float(g.std()):.3f} q01 {float(q[0]):.3f} q50 {float(q[1]):.3f} q99 {float(q[2]):.3f}")
