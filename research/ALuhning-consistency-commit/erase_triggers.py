"""Which tokens trigger the linear-attention 'erase' gate? For every long-memory head (median half-life >= 256
tokens on this text), mark tokens whose half-life is in that head's bottom 5%, and count by token text.
Usage: python erase_triggers.py MODEL_DIR TEXT_FILE"""
import sys, math, collections, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
ids = tok(open(sys.argv[2]).read(), return_tensors="pt").input_ids
layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
caps = {}
for i, l in enumerate(layers):
    if hasattr(l, "linear_attn"):
        l.linear_attn.register_forward_pre_hook(lambda m, a, kw, i=i: caps.__setitem__(i, a[0] if a else kw["hidden_states"]), with_kwargs=True)
with torch.no_grad():
    model(ids)
toks = [tok.decode([t]) for t in ids[0]]
count = collections.Counter(); heads = 0; per_pos = collections.Counter()
for i, x in caps.items():
    la = layers[i].linear_attn
    a = la.in_proj_a(x[0]).float()
    hl = math.log(0.5) / (-(la.A_log.float().exp()) * F.softplus(a + la.dt_bias.float()))
    for h in (hl.median(0).values >= 256).nonzero().flatten().tolist():
        heads += 1; thr = hl[:, h].quantile(0.05)
        for p in (hl[:, h] <= thr).nonzero().flatten().tolist():
            count[repr(toks[p])] += 1; per_pos[p] += 1
print(f"{heads} long-memory heads, {len(toks)} tokens; expected count per token if random = {heads*0.05:.1f}")
print("most frequent reset tokens (token: heads firing, summed over occurrences):")
for t, c in count.most_common(25): print(f"  {t:20s} {c}")
print("positions where the most heads reset at once:")
for p, c in per_pos.most_common(15): print(f"  pos {p:4d} {c:3d} heads  ...{''.join(toks[max(0,p-6):p])}[[{toks[p]}]]{''.join(toks[p+1:p+3])}".replace("\n", "\\n"))
