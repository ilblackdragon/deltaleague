"""C1: fdad0374 + roshaans' continuation delta (0c834b41 - d5fc95ea), added only where it does not
contradict the sign of our merged delta (ac147fa6 - 9492e219).

fb984a19 (SpacexCeo) adds the whole continuation delta. Here, per element:
  - our delta is zero                -> add the continuation change
  - same sign as our delta           -> add it
  - opposite sign to our delta       -> drop it (in the 2-run analysis, about half of co-located moves
                                        disagree in sign, mostly bf16 optimiser noise)
The final norm and the layer-23 graft tensors (where fdad0374 differs from ac147fa6) keep fdad0374's values.
Also reports how far fb984a19 is from a plain fdad0374 + delta, as a check on that recipe.

usage: c1_merge.py BASE_9492 AC147 FDAD D5FC 0C83 FB98 OUT"""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
paths = sys.argv[1:7]; out_p = sys.argv[7]
B, AC, FD, D5, C8, FB = [safe_open(p, 'pt') for p in paths]
NORM = 'model.language_model.norm.weight'
CHUNK = 8_000_000
out, st = {}, dict(graft=[], added=0, dropped=0, tensors=0, fb_plain_match=0, fb_total=0)
for k in FD.keys():
    fd = FD.get_tensor(k)
    if k == NORM or not torch.equal(fd, AC.get_tensor(k)):
        out[k] = fd
        if k != NORM: st['graft'].append(k)
        continue
    d5, c8 = D5.get_tensor(k), C8.get_tensor(k)
    if torch.equal(d5, c8):
        out[k] = fd; continue
    b, fb = B.get_tensor(k), FB.get_tensor(k)
    res = torch.empty_like(fd)
    # float64 work in chunks of CHUNK elements keeps peak RAM low (the embedding has 254M elements)
    for f in (fd, d5, c8, b, fb, res):
        assert f.is_contiguous()
    vfd, vd5, vc8, vb, vfb, vres = (t.view(-1) for t in (fd, d5, c8, b, fb, res))
    for i in range(0, vfd.numel(), CHUNK):
        sl = slice(i, i + CHUNK)
        dr = vc8[sl].double() - vd5[sl].double()
        ours = vfd[sl].double() - vb[sl].double()
        ok = (dr != 0) & ((ours == 0) | (torch.sign(dr) == torch.sign(ours)))
        st['added'] += ok.sum().item(); st['dropped'] += ((dr != 0) & ~ok).sum().item()
        vres[sl] = (vfd[sl].double() + torch.where(ok, dr, torch.zeros_like(dr))).to(fd.dtype)
        st['fb_plain_match'] += ((vfd[sl].double() + dr).to(fd.dtype) == vfb[sl]).sum().item()
    st['fb_total'] += fd.numel(); st['tensors'] += 1
    out[k] = res
    del d5, c8, b, fb
save_file(out, out_p, metadata={'format': 'pt'})
print({**st, 'fb_plain_match_rate': st['fb_plain_match'] / max(st['fb_total'], 1)})
