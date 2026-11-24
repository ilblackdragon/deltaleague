"""Breakage check: mean token NLL of fixed correct answers (c8's own probe answers, under 900 tokens) under several
weight files that share the parent's config. usage: python nll_check.py PARENT_DIR HOLD.jsonl W1 [W2 ...]"""
import json, sys, torch
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer
pdir, hold, ws = sys.argv[1], sys.argv[2], sys.argv[3:]
tok = AutoTokenizer.from_pretrained(pdir)
rows = [json.loads(l) for l in open(hold, encoding='utf-8')]
rows = [r for r in rows if r['score'] == 1 and r['tokens'] < 900][:16]
ids = [torch.tensor([tok(r['text'], add_special_tokens=False)['input_ids']]) for r in rows]
model = AutoModelForImageTextToText.from_pretrained(pdir, dtype=torch.float32).eval()
for w in ws:
    sd = load_file(w); model.load_state_dict(sd, strict=False)
    tot = n = 0
    with torch.no_grad():
        for x in ids:
            l = model(input_ids=x, labels=x).loss.item(); tot += l * (x.shape[1] - 1); n += x.shape[1] - 1
    print(f'{w}: mean NLL {tot / n:.4f} over {n} tokens', flush=True)
