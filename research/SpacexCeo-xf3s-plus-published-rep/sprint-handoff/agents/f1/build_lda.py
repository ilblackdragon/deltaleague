"""F1 builder: add a hidden-state discriminant direction to the </think> output/input row on top of a parent checkpoint.
usage: build_lda.py PARENT_DIR OUT_DIR direction.json delta
  e_close <- e_close + delta * d_hat   (d_hat unit vector in the final-norm output space; logit change = delta * (h . d_hat))
Keeps every other tensor identical; copies tokenizer/config files; prints sha256 and the row norms."""
import sys, os, json, shutil, hashlib, torch
from safetensors.torch import load_file, save_file
parent, out, dfile, delta = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
W = load_file(os.path.join(parent, 'model.safetensors'))
EMB = 'model.language_model.embed_tokens.weight'; CLOSE = 248069
d = torch.tensor(json.load(open(dfile))['d'], dtype=torch.float32); d = d / d.norm()
E = W[EMB]; e = E[CLOSE].float(); print('|e_close| before', e.norm().item(), 'cos(e,d)', torch.cosine_similarity(e, d, dim=0).item())
E[CLOSE] = (e + delta * d).to(E.dtype); print('|e_close| after', E[CLOSE].float().norm().item())
W[EMB] = E
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
print('sha256', hashlib.sha256(open(os.path.join(out, 'model.safetensors'), 'rb').read()).hexdigest())
