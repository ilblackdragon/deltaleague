"""Pick training traces: per prompt, among samples that are correct, closed after >= 50 thinking tokens and not truncated,
keep the shortest (--pick short) or the median-length one (--pick median); --math_per N keeps N for gsm8k/math. usage: select_rft.py OUT.jsonl IN1.jsonl [IN2 ...] [--per 1] [--max_tokens 6000]"""
import json, sys, collections
args = [a for a in sys.argv[1:] if not a.startswith('--')]
per = int(sys.argv[sys.argv.index('--per') + 1]) if '--per' in sys.argv else 1
mx = int(sys.argv[sys.argv.index('--max_tokens') + 1]) if '--max_tokens' in sys.argv else 6000
pick = sys.argv[sys.argv.index('--pick') + 1] if '--pick' in sys.argv else 'short'
math_per = int(sys.argv[sys.argv.index('--math_per') + 1]) if '--math_per' in sys.argv else per
args = [a for a in args if not a.isdigit() and a not in ('short', 'median')]
out, ins = args[0], args[1:]
by = collections.defaultdict(list); stats = collections.Counter()
for f in ins:
    for l in open(f):
        r = json.loads(l); stats['gens'] += 1; stats[r['src']] += 1
        if r['score'] == 1 and r['closed'] and not r['truncated'] and r['think_tokens'] >= 50 and r['tokens'] <= mx:
            by[r['id']].append(r)
keep = []
for i, rs in by.items():
    rs.sort(key=lambda r: r['tokens']); n = math_per if rs[0]['src'] in ('gsm8k_train', 'math_train', 'gsm8k', 'math') else per
    if pick == 'median': m = len(rs) // 2; rs = rs[m:] + rs[:m][::-1]
    keep += rs[:n]
src = collections.Counter(r['src'] for r in keep)
with open(out, 'w') as f:
    for r in keep: f.write(json.dumps({'id': r['id'], 'src': r['src'], 'prompt': r['rendered_prompt'], 'completion': r['text']}) + '\n')
print('gens', stats['gens'], 'prompts with a keeper', len(by), 'kept', len(keep), dict(src),
      'mean tokens', round(sum(r['tokens'] for r in keep) / max(1, len(keep))))
