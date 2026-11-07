"""Close-position margins on b8af1dbc (c_xf3s traces teacher-forced up to and including the emitted </think>).
For a close factor f (row </think> scaled by f on top of b8af1dbc's x1.3): p_close at the ACTUAL close position, and the
expected number of closes at earlier positions. Groups: answer-overflow (closed then truncated), closed-wrong, closed-correct."""
import torch, sys, json, collections, statistics
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model'); TH=tok.convert_tokens_to_ids('</think>')
D=torch.load(sys.argv[1]); j=D['cand'].index('</think>')
rows={r['id']:r for r in json.load(open('/home/user/delta/agents/a2/results/c_xf3s.json'))['rows']}
F=[0.85,0.9,0.92,0.95,0.97,1.0,1.03]
def pclose(tv_others, z):
    v=torch.cat([tv_others, z.view(-1,1)],1); top,idx=v.topk(20,1); p=torch.softmax(top,1); cs=p.cumsum(1); p=p*((cs-p)<0.95); p=p/p.sum(1,keepdim=True)
    return (p*(idx==v.shape[1]-1)).sum(1)
G=collections.defaultdict(lambda: collections.defaultdict(list))
for t in D['traces']:
    r=rows[t['id']]; k=t['think_end']
    if k<0: continue
    g='answer_overflow' if r['truncated'] else ('closed_correct' if r['score']>0 else 'closed_wrong')
    ti=t['topi'][:k+1].long(); tv=t['topv'][:k+1].float(); z=t['cz'][:k+1,j].float()
    others=tv.masked_fill(ti==TH,-1e9)
    G[g]['len'].append(k)
    G[g]['margin'].append((z[k]-others[k].max()).item())
    for f in F:
        pc=pclose(others, z*f)
        G[g][f'pk{f}'].append(pc[k].item()); G[g][f'early{f}'].append(pc[:k].sum().item())
print('think_end tokens median by group:',{g:statistics.median(v['len']) for g,v in G.items()},'n:',{g:len(v['len']) for g,v in G.items()})
print('logit margin z_think - best other at the actual close position (f=1): quantiles 10/50/90%')
for g,v in G.items():
    m=torch.tensor(v['margin']); print(f'   {g:16} {[round(m.quantile(q).item(),2) for q in (0.1,0.5,0.9)]}')
print('\nmean p(close at actual close position) and [share of these closes kept = mean p_k(f)/p_k(1)]; E[#earlier closes]')
for g,v in G.items():
    base=torch.tensor(v['pk1.0'])
    line=f'  {g:16}'
    for f in F:
        pk=torch.tensor(v[f'pk{f}']); kept=(pk/base.clamp_min(1e-6)).clamp(max=10).mean().item()
        line+=f' | f={f}: p={pk.mean():.2f} kept={kept:.2f} early={torch.tensor(v[f"early{f}"]).mean():.2f}'
    print(line)
print('\nbootstrap 90% CI of "kept" share (2000 resamples of traces)')
gen=torch.Generator().manual_seed(0)
for g,v in G.items():
    base=torch.tensor(v['pk1.0']).clamp_min(1e-6); line=f'  {g:16} n={len(base)}'
    for f in [0.9,0.92,0.95,0.97]:
        r=(torch.tensor(v[f'pk{f}'])/base).clamp(max=10); idx=torch.randint(0,len(r),(2000,len(r)),generator=gen)
        m=r[idx].mean(1); line+=f' | f={f}: {r.mean():.2f} [{m.quantile(0.05):.2f},{m.quantile(0.95):.2f}]'
    print(line)
