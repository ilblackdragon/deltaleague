import json, sys, re, statistics as st
arms = sys.argv[1:]
HES = re.compile(r'\b(Wait|wait|Hmm|hmm|Alternatively|Actually|But wait)\b')
def load(p):
    L=[json.loads(l) for l in open(p)]; return L[0], L[1:]
print(f"{'arm':28} {'n':>3} {'acc':>6} {'gsm8k':>6} {'math':>6} {'arc':>6} {'trunc':>6} {'</think>':>8} {'tok':>6} {'WaitHmm/1k':>10} {'hesit/1k':>9}")
for p in arms:
    m,R=load(p)
    by=lambda b:[r['score'] for r in R if r['bench']==b]
    tot=sum(r['tokens'] for r in R)
    f=lambda xs: f"{100*st.mean(xs):6.1f}" if xs else '   n/a'
    print(f"{p.split('/')[-1]:28} {len(R):3} {f([r['score'] for r in R])} {f(by('gsm8k'))} {f(by('math500'))} {f(by('arc_challenge'))} "
          f"{f([r['truncated'] for r in R])} {f([r['closed_think'] for r in R])} {tot/len(R):6.0f} {1000*sum(r['filler'] for r in R)/tot:10.2f} "
          f"{1000*sum(len(HES.findall(r['text'])) for r in R)/tot:9.2f}")
