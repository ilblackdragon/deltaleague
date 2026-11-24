import sys, json, torch
from safetensors import safe_open
from delta_cli.artifacts import reconstruct
parent, sub, edited = sys.argv[1:4]
m = json.load(open(f'{sub}/manifest.json'))
res = reconstruct(f'{parent}/model.safetensors', f'{sub}/weights.safetensors', m)
bad = 0
with safe_open(edited, 'pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if not torch.equal(t.view(torch.uint8) if t.dtype==torch.float8_e4m3fn else t, res[k].view(torch.uint8) if res[k].dtype==torch.float8_e4m3fn else res[k]): bad += 1; print('MISMATCH', k)
print('tensors', len(res), 'mismatches', bad)
