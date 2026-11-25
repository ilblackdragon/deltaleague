"""Measure the input-dependent forgetting gate of Qwen3.5 linear-attention layers on a text.
decay_log(t) = -exp(A_log) * softplus(a_t + dt_bias), a_t = in_proj_a(x_t). Half-life = ln(0.5)/decay_log.
Compares the parent with two candidate edits for long-memory heads (median half-life >= MIN_HL):
  (A) A_log += ln(0.5)                         -> halves decay on every token, including 'reset' tokens
  (B) dt_bias -= delta (per head, chosen so the MEDIAN token's decay halves) -> quiet tokens remember longer,
      tokens where the model signals a reset (large a_t) still forget sharply.
Usage: python gate_probe.py MODEL_DIR TEXT_FILE"""
import sys, math, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
MIN_HL = 256.0
tok = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
text = open(sys.argv[2]).read()
ids = tok(text, return_tensors="pt").input_ids
layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
caps = {}
for i, l in enumerate(layers):
    if hasattr(l, "linear_attn"):
        l.linear_attn.register_forward_pre_hook(lambda m, args, kw, i=i: caps.__setitem__(i, args[0] if args else kw["hidden_states"]), with_kwargs=True)
with torch.no_grad():
    model(ids)
    rows = []
    for i, x in caps.items():
        la = layers[i].linear_attn
        a = la.in_proj_a(x[0]).float()                     # [T, H]
        A, dt = la.A_log.float(), la.dt_bias.float()
        hl = lambda A_, dt_: math.log(0.5) / (-(A_.exp()) * F.softplus(a + dt_))
        h0 = hl(A, dt)
        med = h0.median(0).values
        for h in (med >= MIN_HL).nonzero().flatten().tolist():
            s = a[:, h] + dt[h]
            m = s.median()
            # delta solving softplus(m - delta) = 0.5 * softplus(m)
            target = 0.5 * F.softplus(m); delta = (m - torch.log(torch.expm1(target))).item()
            hA = hl(A + torch.where(torch.arange(len(A)) == h, math.log(0.5), 0.0), dt)[:, h]
            dtB = dt.clone(); dtB[h] -= delta; hB = hl(A, dtB)[:, h]
            q = lambda v: (v.median().item(), v.quantile(0.05).item())
            rows.append((i, h, delta, q(h0[:, h]), q(hA), q(hB)))
    print(f"tokens={ids.shape[1]}  long-memory heads={len(rows)}  (median half-life >= {MIN_HL:.0f})")
    print("layer head  dt_shift | half-life median / at reset tokens (5th pct):  parent  |  A_log x0.5  |  dt_bias shift")
    for i, h, d, p, A_, B_ in rows:
        print(f"{i:5d} {h:4d} {d:8.3f} | {p[0]:8.0f} /{p[1]:6.1f}  | {A_[0]:8.0f} /{A_[1]:6.1f} | {B_[0]:8.0f} /{B_[1]:6.1f}")
