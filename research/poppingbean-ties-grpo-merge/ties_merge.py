"""TIES merge (no trimming) of two fine-tunes A, C of BASE, then copy the x1.05 final-norm tensor from NORM_SRC.
Per element, with dA = A - BASE and dC = C - BASE (fp32):
  - only one changed            -> keep that change in full (soup would halve it; bf16 rounding often erases a half-ulp)
  - both changed, same sign     -> mean of the two (sum would double it, like ExPO, which hurt)
  - both changed, opposite sign -> elected sign = sign(dA + dC); keep only the change with that sign
Computed in float64, stored in each tensor's original dtype (bf16, a few fp32). usage: ties_merge.py BASE A C NORM_SRC OUT"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
base_p, a_p, c_p, norm_p, out_p = sys.argv[1:6]
NORM = 'model.language_model.norm.weight'
B, A, C, N = [safe_open(p, 'pt') for p in (base_p, a_p, c_p, norm_p)]
assert set(B.keys()) == set(A.keys()) == set(C.keys()) == set(N.keys())
out, stats = {}, dict(onlyA=0, onlyC=0, agree=0, conflict=0, changed_tensors=0)
for k in B.keys():
    b, a, c = B.get_tensor(k), A.get_tensor(k), C.get_tensor(k)
    if k == NORM:
        assert torch.equal(a, b) and torch.equal(c, b), 'both GRPO models must keep the base final norm'
        n = N.get_tensor(k); assert not torch.equal(n, b); out[k] = n; stats['changed_tensors'] += 1; continue
    if torch.equal(a, b) and torch.equal(c, b): out[k] = b; continue
    assert b.dtype in (torch.bfloat16, torch.float32), (k, b.dtype)
    bf, da, dc = b.double(), a.double() - b.double(), c.double() - b.double()
    ma, mc = da != 0, dc != 0
    agree = ma & mc & (torch.sign(da) == torch.sign(dc))
    conflict = ma & mc & ~agree
    elected = torch.sign(da + dc)
    keep_a = ma & (~mc | (conflict & (torch.sign(da) == elected)))
    keep_c = mc & (~ma | (conflict & (torch.sign(dc) == elected)))
    d = torch.zeros_like(bf)
    d = torch.where(agree, (da + dc) / 2, d)
    d = torch.where(keep_a & ~agree, da, d)
    d = torch.where(keep_c & ~agree, dc, d)
    out[k] = (bf + d).to(b.dtype)
    stats['changed_tensors'] += int(not torch.equal(out[k], b))
    stats['onlyA'] += (ma & ~mc).sum().item(); stats['onlyC'] += (mc & ~ma).sum().item()
    stats['agree'] += agree.sum().item(); stats['conflict'] += conflict.sum().item()
save_file(out, out_p, metadata={'format': 'pt'})
print(stats)
