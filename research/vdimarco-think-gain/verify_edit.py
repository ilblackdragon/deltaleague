"""Teacher-forced comparison of parent vs edited checkpoint on the same token sequences.

Checks:
 1. </think> logit shift per position (should be positive everywhere, median near target)
 2. max |logit change| over all other tokens (should be exactly 0 in fp32 math up to bf16 noise)
 3. KL(parent || edited) of the next-token distribution at the position right after a forced
    </think> input token (input-side effect of the tied embedding)
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, set_seed

THINK_END = 248069
parent_dir, edited_weights, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.loads(Path(sys.argv[4]).read_text())
tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)
cfg = AutoConfig.from_pretrained(parent_dir, local_files_only=True)


def load(path):
    m = AutoModelForImageTextToText.from_pretrained(parent_dir, config=cfg, local_files_only=True,
                                                    dtype=torch.float32, use_safetensors=True).eval()
    if path:
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            E = f.get_tensor("model.language_model.embed_tokens.weight")
        m.model.language_model.embed_tokens.weight.data.copy_(E)
        if m.lm_head.weight.data_ptr() != m.model.language_model.embed_tokens.weight.data_ptr():
            m.lm_head.weight.data.copy_(E)
    return m


parent = load(None)
sequences = []
with torch.inference_mode():
    for pi, prompt in enumerate(prompts):
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        set_seed(1000 + pi)
        out = parent.generate(ids, max_new_tokens=96, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
        # force a </think> and a newline so we can measure the input-side effect
        forced = torch.cat([out, torch.tensor([[THINK_END, 198]])], 1)
        sequences.append((ids.shape[1], forced))

    def logits_for(model):
        return [model(input_ids=seq).logits[0].to(torch.float16) for _, seq in sequences]

    lp = logits_for(parent)
    del parent
    import gc; gc.collect()
    edited = load(edited_weights)
    le = logits_for(edited)
    del edited

shifts, other_max, kl_after = [], 0.0, []
for (plen, seq), a, b in zip(sequences, lp, le):
    gen = slice(plen - 1, seq.shape[1] - 3)  # positions predicting thinking tokens
    a, b = a.float(), b.float()
    d = b[gen] - a[gen]
    shifts += d[:, THINK_END].tolist()
    mask = torch.ones(a.shape[1], dtype=torch.bool)
    mask[THINK_END] = False
    other_max = max(other_max, d[:, mask].abs().max().item())
    # position whose input is </think> (predicts the token after it) and the one after
    for pos in (seq.shape[1] - 2, seq.shape[1] - 1):
        p = torch.log_softmax(a[pos], -1)
        q = torch.log_softmax(b[pos], -1)
        kl_after.append(torch.sum(p.exp() * (p - q)).item())
shifts = torch.tensor(shifts)
report = {"positions": len(shifts), "shift_min": shifts.min().item(), "shift_median": shifts.median().item(),
          "shift_max": shifts.max().item(), "shift_all_positive": bool((shifts > 0).all()),
          "max_abs_change_other_tokens": other_max, "kl_after_forced_think_end": kl_after}
Path(out_path).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
