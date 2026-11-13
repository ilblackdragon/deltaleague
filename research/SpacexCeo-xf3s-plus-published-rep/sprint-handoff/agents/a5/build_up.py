"""a5: published sparse 15/19 edit applied to selected layers only, bit-exact upstream values (alpha=1), on top of PARENT.
usage: build_up.py OUT_DIR LAYERS(comma, e.g. 15) [PARENT]"""
import sys, os, json, shutil, hashlib, numpy as np, torch
from safetensors.torch import load_file, save_file
out, layers = sys.argv[1], sys.argv[2].split(',')
parent = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser('~/dl/versions/b8af1dbc-d52b-4bbc-a302-3505d897303c/model')
up = os.path.expanduser('~/dl/upstream'); man = json.load(open(f'{up}/manifest.json'))
W = load_file(os.path.join(parent, 'model.safetensors')); ref = {k: (v.shape, v.dtype) for k, v in W.items()}
log = {}
with np.load(f'{up}/edit.npz', allow_pickle=False) as ed:
    for i, e in enumerate(man['tensors']):
        if e['name'].split('.')[3] not in layers: continue
        o = W[e['name']]; bits = o.view(torch.uint16).numpy().copy()
        assert hashlib.sha256(bits.tobytes()).hexdigest() == e['base_sha256'], 'parent tensor differs from upstream base'
        bits.reshape(-1)[ed[f't{i}_idx']] = ed[f't{i}_val']
        assert hashlib.sha256(bits.tobytes()).hexdigest() == e['edited_sha256']
        W[e['name']] = torch.from_numpy(bits).view(torch.bfloat16); log[e['name']] = e['edited_sha256']
assert {k: (v.shape, v.dtype) for k, v in W.items()} == ref and log
os.makedirs(out, exist_ok=True)
for f in os.listdir(parent):
    if f != 'model.safetensors': shutil.copy(os.path.join(parent, f), os.path.join(out, f))
save_file(W, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
json.dump({'parent': parent, 'layers': layers, 'tensors': log}, open(out.rstrip('/') + '.build.json', 'w'), indent=1)
print('edited', log)
