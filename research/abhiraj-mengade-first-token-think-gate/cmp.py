import json,random,sys
def L(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
fs=sys.argv[1:]; D=[L(f) for f in fs]
for f,d in zip(fs,D):
    v=list(d.values());n=len(v);t=sorted(x['tokens'] for x in v)
    print(f"{f.split('/')[-1]:26} n={n} acc={100*sum(x['score'] for x in v)/n:5.1f} trunc={sum(x['truncated'] for x in v)/n:.3f} tok p50={t[n//2]} mean={sum(t)/n:.0f} wrong_boxed={sum(1 for x in v if 'boxed' in x['text'] and x['score']==0)} spurious2nd={sum(x['text'].count('</think>')>1 for x in v)}")
a=D[0]
for f,b in zip(fs[1:],D[1:]):
    ids=sorted(set(a)&set(b));d=[b[i]['score']-a[i]['score'] for i in ids]
    r=random.Random(0);bs=sorted(sum(r.choices(d,k=len(d)))/len(d) for _ in range(5000))
    print(f"{f.split('/')[-1]} - base: {100*sum(d)/len(d):+.1f} CI[{100*bs[125]:+.1f},{100*bs[4875]:+.1f}] n={len(ids)}")
