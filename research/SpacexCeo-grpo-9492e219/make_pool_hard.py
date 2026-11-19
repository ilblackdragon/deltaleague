"""Harder GRPO pool: math (GSM8K + MATH L1-4) and MMLU-aux/ARC prompts where earlier samples were mixed or failed (incl. truncated),
plus a small share of easy MC for stability. usage: make_pool_hard.py OUT.json"""
import sys, os, json, random, collections
sys.path.insert(0, os.path.expanduser('~/dl/rft')); from gen_rft import build, build_v4
P = {p['id']: p for p in build() + build_v4() if p['scorer'] in ('math', 'mc')}
st = collections.defaultdict(list); tr = collections.defaultdict(list)
for f in ['gen_s0', 'gen_s1', 'g4_s0', 'g4_s1']:
    fn = os.path.expanduser(f'~/dl/rft/{f}.jsonl')
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l); st[r['id']].append(float(r['score'] == 1 and r['closed'] and not r['truncated'])); tr[r['id']].append(r['truncated'])
rng = random.Random(11)
math = [i for i in st if i in P and P[i]['scorer'] == 'math']
mixed_m = [i for i in math if 0 < sum(st[i]) < len(st[i])]
trunc_m = [i for i in math if any(tr[i]) and sum(st[i]) < len(st[i])]
easy_m = [i for i in math if sum(st[i]) == len(st[i])]; rng.shuffle(easy_m)
mc = [i for i in st if i in P and P[i]['scorer'] == 'mc' and P[i]['src'] in ('mmlu_aux', 'arc_train', 'wino_train')]
mixed_mc = [i for i in mc if 0 < sum(st[i]) < len(st[i])]
ids = list(dict.fromkeys(mixed_m + trunc_m + easy_m[:len(mixed_m) // 3] + mixed_mc))
pool = [P[i] for i in ids]
print('pool', len(pool), collections.Counter(p['src'] for p in pool), 'trunc-math', len(trunc_m))
json.dump(pool, open(sys.argv[1], 'w'))
