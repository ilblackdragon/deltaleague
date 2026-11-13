"""Local (advisory) public-validation check of the 9 log-likelihood benchmarks.

Reuses delta_cli.public_eval.text to load the native lm-eval 0.4.13 tasks at the pinned
dataset revisions and to verify the selected public documents by hash, then scores the
exact selected indices with lm-eval's HuggingFace backend instead of a vLLM endpoint.
Primary metrics follow delta_cli.public_eval.suite.PRIMARY.

Usage: python likelihood_eval.py WORKSPACE MODEL_DIR OUTPUT.json [--device cuda] [--batch 16]
"""
import argparse, json, statistics, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from delta_cli.public_eval.text import load_tasks, verify_tasks
from delta_cli.public_eval.suite import PRIMARY

GROUPS = ['arc_challenge', 'arc_easy', 'boolq', 'commonsense_qa', 'hellaswag', 'mmlu', 'openbookqa', 'piqa', 'winogrande']
ap = argparse.ArgumentParser()
ap.add_argument('workspace'); ap.add_argument('model_dir'); ap.add_argument('output')
ap.add_argument('--device', default='cuda'); ap.add_argument('--batch', type=int, default=16)
a = ap.parse_args()
out = Path(a.output); assert not out.exists(), 'refusing to overwrite'
suite = json.loads((Path(a.workspace) / 'validation.json').read_text())
dtype = torch.bfloat16 if a.device == 'cuda' else torch.float32
tok = AutoTokenizer.from_pretrained(a.model_dir)
model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=dtype).to(a.device).eval()
lm = HFLM(pretrained=model, tokenizer=tok, backend='causal', batch_size=a.batch, device=a.device, max_length=4096)
report = {'model_dir': str(a.model_dir), 'device': a.device, 'dtype': str(dtype), 'torch': torch.__version__,
          'suite_sha256': suite['sha256'], 'seed': suite['seed'], 'groups': {}}
for group in GROUPS:
    t0 = time.time()
    spec = suite['text'][group]
    tasks = load_tasks(spec['task'], suite['dataset_revisions'])
    verify_tasks(tasks, spec['selection'])
    res = evaluator.simple_evaluate(model=lm, tasks=list(tasks.values()),
        samples={n: e['indices'] for n, e in spec['selection'].items()}, log_samples=True,
        bootstrap_iters=0, random_seed=suite['seed'], numpy_random_seed=suite['seed'],
        torch_random_seed=suite['seed'], fewshot_random_seed=suite['seed'])
    metric = PRIMARY[group][0]
    items = {f"{task}:{s['doc_id']}": float(s[metric]) for task, rows in res['samples'].items() for s in rows}
    assert set(items) == {f'{t}:{i}' for t, e in spec['selection'].items() for i in e['indices']}
    report['groups'][group] = {'metric': metric, 'score': 100 * statistics.mean(items.values()), 'items': items,
                               'seconds': round(time.time() - t0, 1)}
    print(group, metric, round(report['groups'][group]['score'], 2), f"{time.time()-t0:.0f}s", flush=True)
report['mean_9'] = statistics.mean(g['score'] for g in report['groups'].values())
out.write_text(json.dumps(report, indent=1))
print('mean of 9 likelihood benchmarks', round(report['mean_9'], 3))
