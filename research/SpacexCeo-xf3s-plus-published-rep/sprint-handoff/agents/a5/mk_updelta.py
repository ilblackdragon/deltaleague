import json, numpy as np, torch
from safetensors import safe_open
man = json.load(open('/home/clore/dl/upstream/manifest.json'))
f = safe_open('/home/clore/dl/versions/a9340e14-190b-473a-952f-65722b37d576/model/model.safetensors', 'pt')
D = {}
with np.load('/home/clore/dl/upstream/edit.npz', allow_pickle=False) as ed:
    for i, e in enumerate(man['tensors']):
        o = f.get_tensor(e['name']); bits = o.view(torch.uint16).numpy().copy().reshape(-1)
        bits[ed[f't{i}_idx']] = ed[f't{i}_val']
        new = torch.from_numpy(bits.reshape(o.shape)).view(torch.bfloat16).float()
        D[e['name'].split('.')[3]] = new - o.float()
torch.save(D, 'upstream_delta.pt'); print({k: v.norm().item() for k, v in D.items()})
U = {k: torch.linalg.svd(v, full_matrices=False) for k, v in D.items()}
print('cos(u15,u19)', (U['15'][0][:, 0] @ U['19'][0][:, 0]).item())
torch.save({k: (u[:, 0], s[0], vh[0]) for k, (u, s, vh) in U.items()}, 'upstream_rank1.pt')
