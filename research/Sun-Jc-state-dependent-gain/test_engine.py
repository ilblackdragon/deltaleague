"""Numerical checks for engine.py: batching, padding, cache reordering and re-prefill.

Each check compares next-token logits against a solo, unpadded, unchunked reference run.
usage: python test_engine.py MODEL_DIR
"""
import sys

import torch

from engine import Model, Sampler

model = Model(sys.argv[1])
short = model.encode_prompt("What is 7 + 5?")
long = model.encode_prompt(
    "A train leaves a station at 3 pm travelling at 60 km/h. A second train leaves the same station at 4 pm "
    "travelling at 90 km/h on the same track. At what time does the second train catch up with the first? "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
print(f"prompt lengths: short {len(short)}, long {len(long)}")
STEPS = 12


@torch.inference_mode()
def solo(prompt, steps):
    """Reference: no padding, no mask, single forward for the prompt, greedy decode."""
    out = model.text(input_ids=torch.tensor([prompt]), use_cache=True)
    past, h = out.past_key_values, out.last_hidden_state[:, -1]
    logits, toks = [], []
    for _ in range(steps):
        z = model.head(h)[0]
        logits.append(z)
        t = int(z.argmax())
        toks.append(t)
        out = model.text(input_ids=torch.tensor([[t]]), past_key_values=past, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state[:, -1]
    return torch.stack(logits), toks


ref_s, tok_s = solo(short, STEPS)   # references use the stock transformers attention and cache
ref_l, tok_l = solo(long, STEPS)
model.install_fast_paths()          # everything below runs on the buffered cache + grouped attention
print("fast paths installed; cache layer types:", sorted({type(l).__name__ for l in model.new_cache().layers}))
worst = 0.0


def report(name, got, ref):
    global worst
    d = float((got - ref).abs().max())
    worst = max(worst, d)
    same = int(got.argmax()) == int(ref.argmax())
    print(f"  {name:58s} max|dlogit| {d:.2e}  argmax {'same' if same else 'DIFFERENT'}")
    return same


# 1 + 2: padded batch, prefill chunked so the short row gets all-padding chunks, then batched greedy decode.
print("check 1/2: left-padded batch, chunk=16 prefill, batched decode")
with torch.inference_mode():
    ids_rows = [short, long]
    B, T = 2, len(long)
    from engine import PAD_ID
    ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    for i, r in enumerate(ids_rows):
        ids[i, T - len(r):] = torch.tensor(r)
        mask[i, T - len(r):] = 1
    past = model.new_cache()
    for a in range(0, T, 16):
        b = min(T, a + 16)
        out = model.text(input_ids=ids[:, a:b], attention_mask=mask[:, :b], past_key_values=past, use_cache=True)
        past = out.past_key_values
    h = out.last_hidden_state[:, -1]
    ok = True
    for step in range(STEPS):
        z = model.head(h)
        if step in (0, 5, STEPS - 1):
            ok &= report(f"step {step} short row (padded)", z[0], ref_s[step])
            ok &= report(f"step {step} long row", z[1], ref_l[step])
        toks = torch.tensor([tok_s[step], tok_l[step]])
        if step == 5:
            # 3: drop the long row mid-decode; the short row must be unaffected.
            past.reorder_cache(torch.tensor([0]))
            mask, toks = mask[:1], toks[:1]
            past, mask, h = model.step(toks, past, mask)
            for s2 in range(6, STEPS):
                z = model.head(h)
                if s2 in (6, STEPS - 1):
                    ok &= report(f"step {s2} short row after dropping long row", z[0], ref_s[s2])
                past, mask, h = model.step(torch.tensor([tok_s[s2]]), past, mask)
            break
        past, mask, h = model.step(toks, past, mask)

# 4: re-prefill prompt + generated prefix (as after a suspension) and compare the next-step logits.
print("check 4: re-prefill of prompt + 6 generated tokens (suspend/resume path)")
past, mask, h = model.prefill([short + tok_s[:6], long + tok_l[:6]], chunk=32)
with torch.inference_mode():
    z = model.head(h)
ok &= report("short row resumed at step 6", z[0], ref_s[6])
ok &= report("long row resumed at step 6", z[1], ref_l[6])

# 5: sampler sanity: top-k/top-p/presence penalty on a hand-made distribution.
print("check 5: sampler")
smp = Sampler()
logits = torch.full((1, 50), -30.0)
logits[0, :4] = torch.log(torch.tensor([0.5, 0.3, 0.17, 0.03]))
seen = torch.zeros((1, 50), dtype=torch.bool)
ids_, q, _ = smp.candidates(logits, seen)
kept = {int(i): round(float(p), 4) for i, p in zip(ids_[0], q[0]) if p > 0}
print("  top-p 0.95 keeps tokens with tail mass > 0.05:", kept, "(expect 0,1,2 only: 0.5155/0.3093/0.1753)")
ok &= set(kept) == {0, 1, 2}
seen[0, 0] = True
ids_, q, _ = smp.candidates(logits, seen)
kept = {int(i): round(float(p), 4) for i, p in zip(ids_[0], q[0]) if p > 0}
exp = torch.softmax(torch.log(torch.tensor([0.5, 0.3, 0.17, 0.03])) - torch.tensor([1.5, 0, 0, 0]), -1)
print("  presence penalty 1.5 on token 0:", kept, "| unfiltered expectation", [round(float(x), 4) for x in exp])
draws = torch.tensor([int(smp.sample(logits, seen, torch.tensor([u]))[0]) for u in torch.linspace(0.001, 0.999, 999)])
print("  empirical freq over 999 uniform u:", {int(k): round(float((draws == k).float().mean()), 3) for k in draws.unique()})

print(f"\nworst logit deviation {worst:.2e}; all argmax checks {'PASSED' if ok else 'FAILED'}")
sys.exit(0 if ok and worst < 5e-2 else 1)
