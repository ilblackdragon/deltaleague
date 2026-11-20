"""Gateway / margin statistics: how natural closing works. usage: gateway.py TFDIR RESNAME"""
import sys, numpy as np, json; sys.path.insert(0, '/home/user/delta/agents/f1')
from sampler import *
d, res_name = sys.argv[1], sys.argv[2]
tr = load_dir(d); rows = {r['id']: r for r in json.load(open(f'/home/user/delta/agents/f1/results/{res_name}.json'))['rows']}
CL = 248069
def margin(t, j):
    """z_close - max other raw logit; whether that other is seen; r"""
    others = [(t.topv[j, i], t.topseen[j, i]) for i in range(4) if t.topi[j, i] != CL]
    zo, so = others[0]; zc = t.wz[j, t.widx['</think>']]
    return zc - zo, so, zc / t.topv[j, 0]
print('== actual closes (closed traces): margin z_close - z_topother, top-other seen?, r, zmax, mean logit, prev token')
M = []
for k, t in tr.items():
    if not t.closed: continue
    j = t.n - 1; m, s, r = margin(t, j); M.append((m, s, r))
    print(f"{k:7s} margin {m:+6.1f} other_seen {s!s:5s} r {r:.3f} zmax {t.topv[j,0]:.1f} m_t {t.mlog[j]:+.2f} prev tok {int(t.gid[j-1])} n {t.n}")
if M: M = np.array(M, float); print('margin pct', np.percentile(M[:, 0], [10, 50, 90]).round(1), 'r pct', np.percentile(M[:, 2], [10, 50, 90]).round(3), 'other seen frac', M[:, 1].mean().round(2))
print('== gateway positions (prev token = \\n 198) vs other positions: r distribution, per 1k tokens, in closed vs truncated traces')
for lab, sel in [('closed', lambda t: t.closed), ('truncated', lambda t: not t.closed)]:
    rs_g, rs_o, ng, n = [], [], 0, 0
    for t in tr.values():
        if not sel(t): continue
        r = t.wz[:, t.widx['</think>']] / t.topv[:, 0]; g = np.zeros(t.n, bool); g[1:] = t.gid[:-1] == 198
        rs_g.append(r[g][:-1] if t.closed else r[g]); rs_o.append(r[~g]); ng += g.sum(); n += t.n
    if n == 0: continue
    rg = np.concatenate(rs_g); ro = np.concatenate(rs_o)
    print(f"{lab:9s} gateways/1k {1000*ng/n:.1f} | r at gateways pct50/90/99 {np.percentile(rg,[50,90,99]).round(2) if len(rg) else None} frac>0.6 {np.mean(rg>0.6):.4f} >0.7 {np.mean(rg>0.7):.4f} >0.77 {np.mean(rg>0.77):.4f} | r elsewhere pct99 {np.percentile(ro,99):.2f} frac>0.6 {np.mean(ro>0.6):.5f} >0.7 {np.mean(ro>0.7):.5f}")
# tail of r over all positions (hazard curve): P(r > x)
allr = np.concatenate([(t.wz[:, t.widx['</think>']] / t.topv[:, 0])[:-1] if t.closed else (t.wz[:, t.widx['</think>']] / t.topv[:, 0]) for t in tr.values()])
print('== tail of r over all non-close positions (N=%d): x -> per-1k-token count with r>x' % len(allr))
print({x: round(1000 * np.mean(allr > x), 3) for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]})
