import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

base, out = Path(sys.argv[1]), Path(sys.argv[2])
parts = [(Path(p), float(w)) for p, w in (arg.split("=") for arg in sys.argv[3:])]
handles = [(safe_open(str(p), "pt"), w) for p, w in parts]
tensors = {}
with safe_open(str(base), "pt") as fb:
    for k in fb.keys():
        b = fb.get_tensor(k)
        acc = b.float()
        for h, w in handles:
            acc = acc + w * (h.get_tensor(k).float() - b.float())
        tensors[k] = acc.to(b.dtype)
out.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out), metadata={"format": "pt"})
print(f"{out}: base + " + " + ".join(f"{w}*({p.name} - base)" for p, w in parts))
