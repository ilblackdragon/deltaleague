"""Keep, per prompt, the teacher's shortest correct trace that closes </think> exactly once and fits the cap.
usage: python select_traces.py teacher.jsonl train.jsonl [--max-tokens 4096]"""
import argparse, collections, json
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out')
ap.add_argument('--max-tokens', type=int, default=4096); a = ap.parse_args()
rows = [json.loads(l) for l in open(a.src)]
acc, kept_by = collections.defaultdict(list), collections.Counter()
kept = []
for r in rows:
    fam = r.get('src', '').split('_')[0]
    acc[fam] += [s['score'] for s in r['samples']]
    ok = [s for s in r['samples'] if s['score'] == 1 and not s['truncated'] and s['text'].count('</think>') == 1
          and s['tokens'] <= a.max_tokens]
    if ok:
        best = min(ok, key=lambda s: s['tokens'])
        kept.append({'id': r['id'], 'src': r.get('src'), 'prompt': r['prompt'], 'completion': best['text'], 'tokens': best['tokens']})
        kept_by[fam] += 1
with open(a.out, 'w') as f:
    for k in kept:
        f.write(json.dumps(k) + '\n')
n = sum(len(r['samples']) for r in rows); tr = sum(s['truncated'] for r in rows for s in r['samples'])
print(f'{len(rows)} prompts, {n} samples, teacher truncated {tr / max(n, 1):.1%}, kept {len(kept)}, '
      f'mean kept tokens {sum(k["tokens"] for k in kept) / max(len(kept), 1):.0f}')
for fam, v in sorted(acc.items()):
    print(f'  {fam:10s} teacher acc {sum(v) / len(v):.1%}  kept {kept_by[fam]}')
