import sys, numpy as np, json; sys.path.insert(0,'/home/user/delta/agents/f1')
from sampler import *
tr = load_dir('/home/user/delta/agents/f1/tf/c_new')
rows = {r['id']: r for r in json.load(open('/home/user/delta/agents/f1/results/c_new.json'))['rows']}
H7ids = None
print('n traces', len(tr))
print('calibration under the generating edit (newleader): sum p_close vs closed; sum p_h7 vs actual h7 count; mean log p_act')
tot = dict(pc=0, closed=0, ph7=0, nh7=0, ph13=0, nh13=0, lp=[], out=0, n=0)
for k, t in tr.items():
    o = t.probs(NEWLEADER); wid7 = [t.wid[t.widx[x]] for x in H7]; wid13 = [t.wid[t.widx[x]] for x in H13]
    nh7 = np.isin(t.gid, wid7).sum(); nh13 = np.isin(t.gid, wid13).sum()
    r = rows[int(k.split('_')[0])]
    # premature: positions where close prob high but model continued
    print(f"{k:7s} {r['bench']:8s} n {t.n:5d} closed {t.closed!s:5s} score {r['score']:.0f} | sum p_close {o['p_close'].sum():.2f} max {o['p_close'][:-1].max() if t.n>1 else 0:.2f} | p_h7 {o['p_h7'].sum():.1f} act {nh7:3d} | p_h13 {o['p_h13'].sum():.1f} act {nh13:3d} | subs {o['p_subs'].sum():.1f} | mean log p_act {np.log(np.maximum(o['p_act'],1e-12)).mean():.3f} out-of-nucleus {np.mean(o['p_act']==0):.4f}")
    tot['pc'] += o['p_close'].sum(); tot['closed'] += t.closed; tot['ph7'] += o['p_h7'].sum(); tot['nh7'] += nh7; tot['ph13'] += o['p_h13'].sum(); tot['nh13'] += nh13; tot['out'] += (o['p_act']==0).sum(); tot['n'] += t.n
print('TOTAL sum p_close %.2f vs closed %d ; p_h7 %.1f vs %d ; p_h13 %.1f vs %d ; out-of-nucleus frac %.4f' % (tot['pc'], tot['closed'], tot['ph7'], tot['nh7'], tot['ph13'], tot['nh13'], tot['out']/tot['n']))
# distribution of z_close relative to zmax, and mean logit
zc = np.concatenate([t.wz[:, t.widx['</think>']] for t in tr.values()]); zm = np.concatenate([t.topv[:, 0] for t in tr.values()]); ml = np.concatenate([t.mlog for t in tr.values()])
print('z_close pct', np.percentile(zc, [1, 10, 50, 90, 99, 99.9]).round(1)); print('zmax pct', np.percentile(zm, [1, 10, 50, 90, 99]).round(1)); print('mean logit m_t pct', np.percentile(ml, [1, 10, 50, 90, 99]).round(2))
r = zc / zm; print('r=z_close/zmax pct', np.percentile(r, [50, 90, 99, 99.9]).round(3), 'frac r>0.6', np.mean(r > 0.6).round(4), 'r>0.7', np.mean(r > 0.7).round(4), 'r>0.77', np.mean(r > 0.77).round(4), 'r>0.87', np.mean(r>0.87).round(4))
# at actual closes: r and competitor
for k, t in tr.items():
    if t.closed:
        j = t.n - 1; o = t.probs(BASE); on = t.probs(NEWLEADER)
        top2 = [(t.topi[j, i], round(t.topv[j, i], 1), bool(t.topseen[j, i])) for i in range(3)]
        print(k, 'at close: z_close %.1f zmax %.1f r %.2f  p_close base %.3f newleader %.3f  top3 (id,z,seen)' % (t.wz[j, t.widx['</think>']], t.topv[j, 0], r_:=t.wz[j, t.widx['</think>']]/t.topv[j,0], o['p_close'][j], on['p_close'][j]), top2)
