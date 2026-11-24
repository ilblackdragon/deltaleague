import sys, torch, json
sys.argv=[sys.argv[0], sys.argv[1]] + sys.argv[2:]
exec(open('/home/user/delta/agents/a2/anal.py').read().split("if __name__")[0])
traces=[prep(t) for t in D['traces']]
import collections
print('traces',collections.Counter('trunc' if t['truncated'] else ('ok' if t['score']>0 else 'bad') for t in traces))
EDITS=json.loads(sys.argv[2])
show=sys.argv[3].split('|')
base=run(traces,{})
for name,ed in EDITS.items():
    e={cand.index(k):v for k,v in ed.items()}
    a=run(traces,e)
    print(f'\n### {name}: {ed}')
    for g in ['trunc','ok','bad','trunc_postgold','ok_postgold']:
        if g not in base: continue
        n=base[g+'_n'][0].item() if (g+'_n') in base else 0
        tgt=[cand.index(k) for k in ed]
        b_set=base[g][tgt].sum().item(); a_set=a[g][tgt].sum().item()
        line=f'  [{g:15}] pos {int(n):6d} | edited set exp. count {b_set:7.1f} -> {a_set:7.1f} ({100*(a_set-b_set)/max(b_set,1e-9):+.0f}%)'
        for s in show:
            j=cand.index(s); line+=f' | {s!r} {base[g][j].item():.2f}->{a[g][j].item():.2f}'
        print(line)
