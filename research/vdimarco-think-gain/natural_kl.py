"""KL(parent || edited) at the positions following a NATURAL </think> in a parent generation.

Usage: natural_kl.py PARENT_DIR EDITED_WEIGHTS OUT_JSON PROMPT_JSON_FILE MAX_NEW
Generates with the parent (seed 1000, server sampling settings) until </think> appears or the
budget ends, then teacher-forces the same tokens through parent and edited and reports KL at
the positions after </think> (the answer tokens), plus the </think> logit shift at the
position that emitted it.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, set_seed

THINK_END = 248069
parent_dir, edited_weights, out_path, prompts_path, max_new = sys.argv[1:6]
prompts = json.loads(Path(prompts_path).read_text())
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
seqs = []
with torch.inference_mode():
    for pi, prompt in enumerate(prompts):
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
        set_seed(1000)
        out = parent.generate(ids, max_new_tokens=int(max_new), do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                              eos_token_id=[248044, 248046])
        gen = out[0, ids.shape[1]:].tolist()
        pos = gen.index(THINK_END) if THINK_END in gen else None
        print(f"prompt {pi}: {len(gen)} tokens, </think> at {pos}", file=sys.stderr)
        seqs.append((ids.shape[1], out, pos))
    def sl(model):
        res = []
        for plen, o, pos in seqs:
            if pos is None:
                res.append(None); continue
            lg = model(input_ids=o).logits[0]
            res.append(lg[plen + pos - 1: plen + pos + 12].float().clone())
        return res
    lp = sl(parent)
    # Swap in the edited embedding in place; the head is tied so it follows automatically.
    from safetensors import safe_open
    with safe_open(edited_weights, "pt") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight")
    emb = parent.model.language_model.embed_tokens.weight
    emb.data.copy_(E)
    if parent.lm_head.weight.data_ptr() != emb.data_ptr():
        parent.lm_head.weight.data.copy_(E)
    del E
    le = sl(parent)

report = []
for (plen, out, pos), a, b in zip(seqs, lp, le):
    if pos is None:
        report.append({"closed": False, "tokens": int(out.shape[1] - plen)})
        continue
    base = plen + pos - 1
    p_emit = 0  # slice starts at the position whose prediction is </think>
    kls = []
    for k in range(0, min(11, out.shape[1] - plen - pos - 1)):
        q_pos = 1 + k  # slice index: input token is </think> (k=0) or later answer tokens
        p = torch.log_softmax(a[q_pos], -1)
        q = torch.log_softmax(b[q_pos], -1)
        kls.append(round(torch.sum(p.exp() * (p - q)).item(), 5))
    report.append({"closed": True, "tokens": int(out.shape[1] - plen), "think_tokens": pos,
                   "think_logit_parent_at_emit": a[p_emit, THINK_END].item(),
                   "think_logit_edited_at_emit": b[p_emit, THINK_END].item(),
                   "max_other_logit_at_emit": a[p_emit].topk(2).values[1].item(),
                   "kl_after_think_end_by_offset": kls,
                   "answer_text": tok.decode(out[0, plen + pos + 1:plen + pos + 13])})
Path(out_path).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
