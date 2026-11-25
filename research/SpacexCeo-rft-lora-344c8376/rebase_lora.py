"""Re-target a Delta LoRA submission at another parent version (same tensors, optional alpha scale) and rebuild the merged model.
usage: rebase_lora.py SUB_IN NEW_PARENT_MODEL_DIR NEW_PARENT_VERSION_ID NEW_PARENT_CONTENT_HASH SUB_OUT MODEL_OUT [SCALE]"""
import sys, json, os, shutil, hashlib
from safetensors.torch import save_file
sin, par, pid, phash, sout, mout = sys.argv[1:7]; s = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
man = json.load(open(os.path.join(sin, 'manifest.json')))
man['parent_version_id'], man['parent_content_hash'] = pid, phash
for t in man['tensors'].values(): t['alpha'] = t['alpha'] * s
os.makedirs(sout, exist_ok=True); shutil.copy(os.path.join(sin, 'weights.safetensors'), os.path.join(sout, 'weights.safetensors'))
json.dump(man, open(os.path.join(sout, 'manifest.json'), 'w'), indent=2)
from delta_cli.artifacts import reconstruct
res = reconstruct(os.path.join(par, 'model.safetensors'), os.path.join(sout, 'weights.safetensors'), man, allowed_prefix='model.language_model.')
os.makedirs(mout, exist_ok=True)
for f in os.listdir(par):
    if f != 'model.safetensors': shutil.copy(os.path.join(par, f), os.path.join(mout, f))
save_file(res, os.path.join(mout, 'model.safetensors'), metadata={'format': 'pt'})
print('rebased onto', pid, 'scale', s, 'manifest sha', hashlib.sha256(open(os.path.join(sout, 'manifest.json'), 'rb').read()).hexdigest())
