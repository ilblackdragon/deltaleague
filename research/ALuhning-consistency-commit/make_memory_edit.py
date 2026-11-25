"""Hand-set arm: double the memory of long-memory linear-attention heads. For every Gated-DeltaNet head whose
median half-life on TEXT_FILE is >= 256 tokens (108 heads on sample_trace.txt), A_log[h] += ln(0.5), which
halves the decay rate on every token. Only *.linear_attn.A_log changes (float32, exact).
Usage: python make_memory_edit.py PARENT_DIR TEXT_FILE OUT_DIR"""
import sys, math, shutil, torch, torch.nn.functional as F
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
parent, text, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(parent)
model = AutoModelForCausalLM.from_pretrained(parent, dtype=torch.float32).eval()
layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
caps = {}
for i, l in enumerate(layers):
    if hasattr(l, "linear_attn"):
        l.linear_attn.register_forward_pre_hook(lambda m, a, kw, i=i: caps.__setitem__(i, a[0] if a else kw["hidden_states"]), with_kwargs=True)
with torch.no_grad():
    model.model(input_ids=tok(open(text).read(), return_tensors="pt").input_ids)
sd = load_file(f"{parent}/model.safetensors")
with safe_open(f"{parent}/model.safetensors", "pt") as f:
    meta = f.metadata()
total = 0
for i, x in caps.items():
    la = layers[i].linear_attn
    hl = math.log(0.5) / (-(la.A_log.float().exp()) * F.softplus(la.in_proj_a(x[0]).float() + la.dt_bias.float()))
    heads = (hl.median(0).values >= 256).nonzero().flatten()
    key = f"model.language_model.layers.{i}.linear_attn.A_log"
    t = sd[key].clone(); t[heads] += math.log(0.5); sd[key] = t; total += len(heads)
    print(f"layer {i}: heads {heads.tolist()}")
shutil.copytree(parent, out, ignore=shutil.ignore_patterns("model.safetensors"))
save_file(sd, f"{out}/model.safetensors", metadata=meta)
print("edited heads:", total)
