"""Independent benchmark execution; no platform API, database or GPU lifecycle."""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil

from .coverage import categories, populations
from .generation import sampling
from .suite import manifest

DATA = Path(__file__).parent / 'data'
TERMINAL = 'terminal_bench_2_1'


def make_protocol(benchmark, generation, *, coverage='sample', concurrency=8,
                  max_model_len=34816, task_ids=None, attempts=1):
    """Declare semantics before execution; generation settings are always explicit."""
    if coverage not in ('sample', 'full'):
        raise ValueError('coverage must be sample or full')
    if type(concurrency) is not int or not 1 <= concurrency <= 256:
        raise ValueError('concurrency must be between 1 and 256')
    if type(generation.get('max_tokens')) is not int or not 0 < generation['max_tokens'] < max_model_len:
        raise ValueError('Explicit generation max_tokens must fit the context window')
    suite = manifest()
    p = {'id': 'public-benchmark-v1', 'benchmarks': [benchmark], 'coverage': coverage,
         'generation': deepcopy(generation), 'thinking': bool(generation.get('enable_thinking')),
         'sampling': sampling(generation), 'max_tokens': generation['max_tokens'],
         'max_model_len': max_model_len, 'concurrency': concurrency,
         'seed': generation.get('seed', suite['seed']), 'suite_sha256': suite['sha256']}
    if benchmark == TERMINAL:
        data = json.loads((DATA / 'terminal_bench_2_1.json').read_text())
        if not generation.get('enable_thinking') or generation['max_tokens'] != 8192:
            raise ValueError('This Terminal-Bench profile requires 8192-token thinking generation')
        if any(key not in generation for key in ('temperature', 'top_p', 'top_k', 'presence_penalty')):
            raise ValueError('Terminal generation requires explicit sampling settings')
        if attempts not in (1, 5):
            raise ValueError('Terminal attempts must be 1 or 5')
        selected = sorted(data['tasks']) if coverage == 'full' else list(task_ids or [])
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(data['tasks']):
            raise ValueError('Sample coverage requires unique native task IDs')
        p.update(terminal_benchmark=True, dataset_revision=data['revision'], task_ids=selected,
                 attempts_per_task=attempts, expected_items=len(selected)*attempts,
                 task_concurrency=concurrency, harbor_version='0.25.0', agent='terminus-2',
                 context_counting='qwen-tokenizer-v1',
                 agent_options={'parser_name': 'json', 'enable_summarize': True,
                                'proactive_summarization_threshold': 8000},
                 terminal_manifest_sha256=hashlib.sha256((DATA/'terminal_bench_2_1.json').read_bytes()).hexdigest())
    elif benchmark in populations(suite):
        if coverage == 'full':
            count = populations(suite)[benchmark]
        elif benchmark in suite['text']:
            count = suite['text'][benchmark]['count']
        elif benchmark in suite['code']:
            count = suite['code'][benchmark]['count']
        else:
            count = sum(suite['bfcl'][c]['count'] for c in categories(suite, benchmark))
        p['expected_items'] = count
    else:
        raise ValueError('Unknown benchmark: '+benchmark)
    return p


def execute(root, phase_callback=None):
    """Run one prepared public benchmark or shard and return native summary metrics.

    The directory contract contains only local protocol/data/cache files and an
    inference endpoint. Callers own process isolation, task leases and publishing.
    """
    root = Path(root)
    config = json.loads((root/'manifest.json').read_text())
    if config['protocol'].get('terminal_benchmark'):
        from .terminal import evaluate
        if version('harbor') != config['protocol']['harbor_version']:
            raise ValueError('Harbor version differs from the declared protocol')
        evaluate(root)
        if phase_callback: phase_callback('scoring')
        return json.loads((root/'summary.json').read_text())
    from .benchmark import evaluate
    from .dataset_mirror import use_sources
    with use_sources(config):
        return evaluate(root, phase_callback=phase_callback)


def run_benchmark(*, protocol, endpoint, model, tokenizer, content_hash, output,
                  bounds=None, api_key_env=None):
    """Standalone execution against a local, self-hosted or external model endpoint.

    Run separate processes for concurrent benchmarks: native harnesses mutate
    process-global registration and tokenizer state. API keys are read only from
    the named environment variable and never written to the run manifest.
    """
    if not re.fullmatch('[0-9a-f]{64}', content_hash):
        raise ValueError('content_hash must be a lowercase SHA-256 identity')
    from urllib.parse import urlsplit
    url = urlsplit(endpoint)
    if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP endpoint without embedded credentials or query parameters')
    p = deepcopy(protocol)
    if len(p.get('benchmarks', [])) != 1:
        raise ValueError('Choose one benchmark per invocation')
    if bounds is not None and (p.get('terminal_benchmark') or len(bounds) != 2
            or any(type(n) is not int for n in bounds) or not 0 <= bounds[0] < bounds[1] <= p['expected_items']):
        raise ValueError('Invalid shard bounds; Terminal selection uses native task IDs')
    p['content_hash'] = content_hash
    root = Path(output)
    tokenizer = Path(tokenizer).resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    (root/'model').symlink_to(tokenizer, target_is_directory=True)
    from .history import History, export_checkpoint
    export_checkpoint(History(root/'empty-cache.sqlite3'), 'public', root/'history-seed.sqlite3.gz')
    (root/'empty-cache.sqlite3').unlink()
    if p.get('terminal_benchmark'):
        shutil.copyfile(DATA/'terminal_bench_2_1.json', root/'terminal-manifest.json')
    config = {'protocol': p, 'protocol_hash': hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
              'inference_upstream': endpoint.rstrip('/').removesuffix('/v1'),
              'inference_model': model, 'api_key_env': api_key_env,
              'task_slice': bounds, 'task_count': bounds[1]-bounds[0] if bounds else p['expected_items']}
    (root/'manifest.json').write_text(json.dumps(config, indent=2))
    result = execute(root)
    (root/'result.json').write_text(json.dumps(result, indent=2))
    return result
