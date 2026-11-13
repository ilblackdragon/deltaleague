import json, sys, math
base = sys.argv[1]
B = {r['id']: r for r in json.load(open(f'data/{base}.json'))['rows']}
for j in sys.argv[2:]:
    d = json.load(open(f'data/{j}.json')); R = {r['id']: r for r in d['rows']}
    s = d['summary']
    diffs = [R[i]['score'] - B[i]['score'] for i in B]
    m = sum(diffs) / len(diffs); sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
    print(f"{j:14s} ALL {s['ALL']['acc']:5.1f} trunc {s['ALL']['trunc']:.3f} tok {s['ALL']['tokens']:5.0f} accC {s['ALL']['acc_closed']:5.1f} hedges {s['ALL']['hedges']:5.1f} | d vs {base}: {100*m:+.1f} +- {100*sd:.1f} (w/l {sum(x>0 for x in diffs)}/{sum(x<0 for x in diffs)}) |",
          ' '.join(f"{b}:{v['acc']:.1f}/{v['trunc']:.2f}" for b, v in s.items() if b != 'ALL'))
