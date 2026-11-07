"""Build the GRPO prompt pool from public TRAIN prompts (gen_rft build v1 + v4, math and multiple-choice only), weighted toward
prompts of intermediate difficulty measured on earlier 344c8376/cd0b95bb samples (rft/gen_s*.jsonl, g4_s*.jsonl).
usage: make_pool.py OUT.json"""
import sys, os, json, random, collections
sys.path.insert(0, os.path.expanduser('~/dl/rft')); from gen_rft import build, build_v4
P = {p['id']: p for p in build() + build_v4() if p['scorer'] in ('math', 'mc')}
st = collections.defaultdict(list)
for f in ['gen_s0', 'gen_s1', 'g4_s0', 'g4_s1']:
    fn = os.path.expanduser(f'~/dl/rft/{f}.jsonl')
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l); st[r['id']].append(float(r['score'] == 1 and r['closed'] and not r['truncated'] and r['think_tokens'] >= 50))
mixed = [i for i, v in st.items() if i in P and 0 < sum(v) < len(v)]
easy = [i for i, v in st.items() if i in P and sum(v) == len(v)]
hard = [i for i, v in st.items() if i in P and sum(v) == 0]
rng = random.Random(7); rng.shuffle(easy); rng.shuffle(hard)
ids = mixed + easy[:len(mixed) // 2] + hard[:len(mixed) // 3]
pool = [P[i] for i in ids]
print('measured', len(st), 'mixed', len(mixed), 'easy', len(easy), 'hard', len(hard), 'pool', len(pool), collections.Counter(p['src'] for p in pool))
json.dump(pool, open(sys.argv[1], 'w'))
