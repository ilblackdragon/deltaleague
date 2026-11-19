"""little Qwen v0.1 as an exact rank-1 LoRA patch for Delta League (11-05): every changed tensor's delta is rank 1.
  projection: W' = W - d (d.W)          -> B = -d [out,1], A = d.W [1,in]
  n998 x -2:  W'[:,998] = -2 W[:,998]   -> B = -3 W[:,998] [out,1], A = e_998 [1,in]
L16 down_proj gets no projection (it is outside layers 0-13), so each tensor has exactly one rank-1 term.
Usage: python make_lora.py PARENT_DIR OUT_DIR [SCALE]   (SCALE multiplies the projection terms; 1.0 = v0.1, 0.85 = submission C)
PARENT_DIR is a `delta model pull` folder (model.safetensors + delta-model.json). Then: delta validate OUT_DIR"""
import os, sys, json, hashlib, torch
from safetensors import safe_open
from safetensors.torch import save_file
parent, out = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
os.makedirs(out, exist_ok=True)
p = safe_open(f"{parent}/model.safetensors", "pt")
d = torch.tensor(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "doubt_direction_L13.json")))["values"], dtype=torch.float32); d /= d.norm()
meta = json.load(open(f"{parent}/delta-model.json"))
tensors, w = {}, {}
for i, (layer, part) in enumerate([(l, q) for l in range(14) for q in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj")]):
    k = f"model.language_model.layers.{layer}.{part}.weight"
    if k not in p.keys(): continue
    W = p.get_tensor(k).float()
    w[f"t{i}.a"] = (d @ W)[None, :].contiguous(); w[f"t{i}.b"] = (-d)[:, None].contiguous()
    tensors[k] = {"a": f"t{i}.a", "b": f"t{i}.b", "alpha": scale}
k = "model.language_model.layers.16.mlp.down_proj.weight"; W = p.get_tensor(k).float()
A = torch.zeros(1, W.shape[1]); A[0, 998] = 1.0
w["n998.a"] = A; w["n998.b"] = (-3 * W[:, 998])[:, None].contiguous()
tensors[k] = {"a": "n998.a", "b": "n998.b", "alpha": 1.0}
save_file(w, f"{out}/weights.safetensors")
data = open(f"{out}/weights.safetensors", "rb").read()
man = {"schema_version": "delta.submission.v1", "format": "lora", "operation": "add", "reconstruction_version": "delta-reconstruct-v1",
       "parent_version_id": meta["id"], "parent_content_hash": meta["content_hash"], "model_config_id": meta["model_id"],
       "tensors": tensors, "artifacts": [{"name": "weights.safetensors", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
json.dump(man, open(f"{out}/manifest.json", "w"), indent=2)
print(len(tensors), "tensors,", len(data), "bytes")
