"""Local (advisory) estimate of the private-test regime: thinking on, sampled decoding.

Items: the public-validation gsm8k selection (lm-eval task docs at the pinned revision,
hash-verified), in ascending index order, first N items. Generation uses the CLI's own
Runner with evaluation.json's settings (T=1.0, top_p=.95, top_k=20, presence 1.5, thinking)
except max_tokens. Base and edit run the same item with the same seed (1000 + position).
Scoring: delta_cli.scoring math scorer (math_verify) on the text after </think>.
Raw generations go to GENERATIONS (keep out of Git); the report holds scores/tokens only.

Usage: python gsm8k_sampled.py WORKSPACE BASE_DIR EDIT_DIR REPORT.json GENERATIONS.jsonl
         [--n 40] [--max-tokens 4096] [--deadline-minutes 100] [--device cuda]
"""
import argparse, json, statistics, time
from pathlib import Path
from delta_cli.local import Runner
from delta_cli.scoring import score_answer
from delta_cli.public_eval.text import load_tasks, verify_tasks

ap = argparse.ArgumentParser()
for name in ['workspace', 'base', 'edit', 'report', 'generations']: ap.add_argument(name)
ap.add_argument('--n', type=int, default=40); ap.add_argument('--max-tokens', type=int, default=4096)
ap.add_argument('--deadline-minutes', type=float, default=100); ap.add_argument('--device', default='cuda')
a = ap.parse_args()
report, gens = Path(a.report), Path(a.generations)
assert not report.exists() and not gens.exists(), 'refusing to overwrite'
ws = Path(a.workspace)
suite = json.loads((ws / 'validation.json').read_text())
generation = dict(json.loads((ws / 'evaluation.json').read_text())['generation'], max_tokens=a.max_tokens)
spec = suite['text']['gsm8k']
tasks = load_tasks(spec['task'], suite['dataset_revisions']); verify_tasks(tasks, spec['selection'])
docs = tasks['gsm8k'].eval_docs
indices = sorted(spec['selection']['gsm8k']['indices'])[:a.n]
runners = {'base': Runner(a.base, a.device), 'edit': Runner(a.edit, a.device)}
deadline = time.monotonic() + 60 * a.deadline_minutes
groups = {'base': {'items': {}}, 'edit': {'items': {}}}; extra = {'base': [], 'edit': []}
for pos, idx in enumerate(indices):
    if time.monotonic() > deadline:
        print('deadline reached; stopping before item', pos, flush=True); break
    doc = docs[idx]
    gold = doc['answer'].split('####')[-1].strip().replace(',', '')
    prompt = doc['question'] + '\nPlease reason step by step, and put your final answer within \boxed{}.'
    for name, runner in runners.items():
        r = runner.generate(prompt, generation, 1000 + pos)
        s = score_answer({'scorer': 'math', 'answer': gold}, r['text'])
        groups[name]['items'][f'gsm8k:{idx}'] = s
        extra[name].append({'tokens': r['tokens'], 'truncated': r['truncated'], 'seconds': r['generation_seconds']})
        with gens.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'model': name, 'doc_id': idx, 'seed': 1000 + pos, 'score': s, **r}) + '\n')
        print(pos, idx, name, s, r['tokens'], 'trunc' if r['truncated'] else '', f"{r['generation_seconds']:.0f}s", flush=True)
for name in groups:
    g = groups[name]; e = extra[name]
    g.update(score=100 * statistics.mean(g['items'].values()), truncation_rate=statistics.mean(x['truncated'] for x in e),
             mean_tokens=statistics.mean(x['tokens'] for x in e), total_seconds=sum(x['seconds'] for x in e))
    out = {'groups': {'gsm8k_sampled': g}, 'generation': generation, 'runtime': runners[name].details,
           'seeds': [1000 + p for p in range(len(g['items']))]}
    Path(str(report).replace('.json', f'.{name}.json')).write_text(json.dumps(out, indent=1))
    print(name, 'score', round(g['score'], 1), 'n', len(g['items']), 'trunc', g['truncation_rate'], 'tokens', round(g['mean_tokens']))
