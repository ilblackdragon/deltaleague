"""Delta statistics of two fine-tunes vs a common base, in bf16 ulps of the base value."""
import sys, torch
from safetensors import safe_open
B, A, C = [safe_open(p, 'pt') for p in sys.argv[1:4]]
def ulp(x):  # bf16 spacing at |x|
    e = torch.floor(torch.log2(x.float().abs().clamp_min(2**-126)))
    return torch.pow(2.0, e - 7)
tot = dict(n=0, a=0, c=0, both=0, agree=0, a1=0, c1=0, abig=0, cbig=0, bothbig=0, agreebig=0)
fam = {}
for k in B.keys():
    b = B.get_tensor(k)
    if b.dtype != torch.bfloat16: continue
    a, c = A.get_tensor(k), C.get_tensor(k)
    if torch.equal(a, b) and torch.equal(c, b): continue
    u = ulp(b); da = (a.float() - b.float()) / u; dc = (c.float() - b.float()) / u
    ma, mc = da != 0, dc != 0; ba, bc = da.abs() > 1.5, dc.abs() > 1.5
    s = dict(n=b.numel(), a=ma.sum().item(), c=mc.sum().item(), both=(ma & mc).sum().item(),
             agree=(ma & mc & (da.sign() == dc.sign())).sum().item(), a1=(da.abs() == 1).sum().item(),
             c1=(dc.abs() == 1).sum().item(), abig=ba.sum().item(), cbig=bc.sum().item(),
             bothbig=(ba & bc).sum().item(), agreebig=(ba & bc & (da.sign() == dc.sign())).sum().item())
    f = 'embed' if 'embed' in k else 'norm' if 'norm' in k else 'mlp' if 'mlp' in k else 'attn'
    for d in (tot, fam.setdefault(f, dict.fromkeys(tot, 0))):
        for x in s: d[x] += s[x]
for name, d in [('ALL', tot)] + sorted(fam.items()):
    n = d['n']
    print(f"{name:6s} n={n:>11,} changedA {d['a']/n:6.2%} changedC {d['c']/n:6.2%} | A 1ulp {d['a1']/max(d['a'],1):5.1%} C 1ulp {d['c1']/max(d['c'],1):5.1%} | "
          f"both {d['both']/n:6.2%} sign-agree {d['agree']/max(d['both'],1):5.1%} | >1ulp A {d['abig']/n:6.3%} C {d['cbig']/n:6.3%} both {d['bothbig']/n:7.4%} agree {d['agreebig']/max(d['bothbig'],1):5.1%}")
