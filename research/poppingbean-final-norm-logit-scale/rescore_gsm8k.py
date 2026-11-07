"""Re-score gsm8k_sampled.py generations in-process.

delta_cli.scoring's math scorer calls math_verify with its default timeouts; on Windows
math_verify implements those timeouts with multiprocessing, which fails here
(OSError WinError 6) and makes every answer score 0. This applies the same rule
(text after </think>, math_verify parse+verify with Latex and Expr extraction) with
timeouts disabled, and writes per-model reports (lenient = CLI rule, where text without
</think> is scored whole; strict = only answers after a closed trace count) in compare_items.py format.

Usage: python rescore_gsm8k.py WORKSPACE GENERATIONS.jsonl REPORT_PREFIX
"""
import json, statistics, sys
from pathlib import Path
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from delta_cli.public_eval.text import load_tasks, verify_tasks

CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
def score(gold, text):
    answer = text.split('</think>')[-1].strip()
    g = parse(str(gold), extraction_config=CFG, parsing_timeout=None)
    p = parse(answer, extraction_config=CFG, parsing_timeout=None)
    return float(bool(g) and bool(p) and verify(g, p, timeout_seconds=None))

if __name__ == '__main__':
    ws, gens, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    suite = json.loads((ws / 'validation.json').read_text())
    spec = suite['text']['gsm8k']
    tasks = load_tasks(spec['task'], suite['dataset_revisions']); verify_tasks(tasks, spec['selection'])
    docs = tasks['gsm8k'].eval_docs
    rows = [json.loads(l) for l in gens.read_text(encoding='utf-8').splitlines() if l.strip()]
    done = {}
    for r in rows: done.setdefault(r['doc_id'], {})[r['model']] = r
    paired = sorted(d for d, m in done.items() if {'base', 'edit'} <= set(m))  # only complete pairs
    for name in ['base', 'edit']:
        items, strict, toks, trunc, closed = {}, {}, [], [], []
        for d in paired:
            r = done[d][name]
            gold = docs[d]['answer'].split('####')[-1].strip().replace(',', '')
            items[f'gsm8k:{d}'] = score(gold, r['text'])
            # Strict: only answers given after a closed thinking trace count.
            strict[f'gsm8k:{d}'] = items[f'gsm8k:{d}'] if '</think>' in r['text'] else 0.0
            toks.append(r['tokens']); trunc.append(r['truncated']); closed.append('</think>' in r['text'])
        g = {'score': 100 * statistics.mean(items.values()), 'items': items, 'n': len(items),
             'mean_tokens': statistics.mean(toks), 'truncation_rate': statistics.mean(trunc),
             'think_closed_rate': statistics.mean(closed)}
        gs = {'score': 100 * statistics.mean(strict.values()), 'items': strict, 'n': len(strict)}
        Path(f'{prefix}.{name}.json').write_text(json.dumps({'groups': {'gsm8k_sampled': g}}, indent=1))
        Path(f'{prefix}-strict.{name}.json').write_text(json.dumps({'groups': {'gsm8k_sampled_strict': gs}}, indent=1))
        print(name, {k: v for k, v in g.items() if k != 'items'}, 'strict', gs['score'])
