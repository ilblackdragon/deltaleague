"""Language-model loss on the modded-nanogpt FineWeb validation window.

The NanoGPT speedrun reports cross-entropy over the first 10,485,760 GPT-2 tokens
of ``fineweb_val_000000.bin`` (``kjj0/fineweb10B-gpt2``). That stream is a sequence
of documents, each introduced by ``<|endoftext|>``. GPT-2 byte-level BPE decodes
losslessly, so the same text is rescored here with the candidate's own tokenizer.

Each document is scored as ``<eot> doc <eot>``: every document token and the
closing end-of-text are predicted, as in the speedrun. The final document is cut
by the window and predicts no closing token. Documents longer than the scoring
window are split into independent windows, each restarting from ``<eot>``.

``loss_per_gpt2_token`` divides the total negative log-likelihood (nats) by the
10,485,759 GPT-2 targets in the window, which puts tokenizer-independent text
likelihood on the speedrun's scale. ``bits_per_byte`` and per-candidate-token
loss are reported alongside it. Lower is better for every metric.
"""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

NAME = 'fineweb_nanogpt_val'
SOURCE = {'repo': 'kjj0/fineweb10B-gpt2', 'revision': '889765ea1f903759787add96995d81171b632d0c',
          'file': 'fineweb_val_000000.bin',
          'sha256': '5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640'}
GPT2_VOCAB = {'repo': 'openai-community/gpt2', 'revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
              'file': 'vocab.json',
              'sha256': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'}
WINDOW_TOKENS = 10_485_760
GPT2_EOT = 50256
DOCUMENTS = 14_983
DOCUMENTS_SHA256 = '367747ab7ec359cdd002520aba84ad2af8fd3e392f14bca5161a027d8ffbf771'
GPT2_TARGETS = WINDOW_TOKENS - 1
UTF8_BYTES = 46_515_251
SCORING_WINDOW = 32768
END_OF_TEXT = '<|endoftext|>'
REDUCE = 'prompt_logprob_sum'


def protocol_fields():
    """Declared, hashable identity of the dataset and scoring rule."""
    return {'metric_kind': 'loss', 'lower_is_better': True, 'primary_metric': 'loss_per_gpt2_token',
            'fineweb_source': SOURCE, 'gpt2_vocab': GPT2_VOCAB, 'window_tokens': WINDOW_TOKENS,
            'documents': DOCUMENTS, 'documents_sha256': DOCUMENTS_SHA256,
            'scoring_window_tokens': SCORING_WINDOW, 'document_framing': 'eot-doc-eot-v1',
            'expected_items': DOCUMENTS}


def _byte_decoder():
    # GPT-2's reversible byte <-> printable-unicode table.
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    chars, extra = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, chars)}


def _verified_download(spec, cache):
    from huggingface_hub import hf_hub_download
    kind = 'dataset' if spec is SOURCE else 'model'
    path = Path(hf_hub_download(spec['repo'], spec['file'], repo_type=kind, revision=spec['revision'],
                                cache_dir=str(cache / 'hf')))
    if hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != spec['sha256']:
        raise ValueError('Pinned FineWeb source changed: ' + spec['file'])
    return path


def build_documents(target):
    """Rebuild docs.jsonl from the pinned shard; the result must match DOCUMENTS_SHA256."""
    import numpy as np
    target = Path(target)
    cache = target.parent
    shard = _verified_download(SOURCE, cache)
    vocab = json.loads(_verified_download(GPT2_VOCAB, cache).read_text())
    decoder, bytes_of = _byte_decoder(), {}
    for token, index in vocab.items():
        bytes_of[index] = bytes(decoder[c] for c in token)
    header = np.fromfile(shard, dtype=np.int32, count=256)
    if header[0] != 20240520 or header[1] != 1:
        raise ValueError('Unexpected FineWeb shard header')
    window = np.fromfile(shard, dtype=np.uint16, count=WINDOW_TOKENS, offset=1024).tolist()
    if len(window) != WINDOW_TOKENS or window[0] != GPT2_EOT:
        raise ValueError('FineWeb window does not start at a document boundary')
    starts = [i for i, t in enumerate(window) if t == GPT2_EOT] + [WINDOW_TOKENS]
    temp = target.with_suffix('.tmp')
    with temp.open('w') as out:
        for i, (a, b) in enumerate(zip(starts, starts[1:])):
            ids = window[a + 1:b]
            raw = b''.join(bytes_of[t] for t in ids)
            truncated = b == WINDOW_TOKENS
            text = raw.decode('utf-8', errors='replace' if truncated else 'strict')
            out.write(json.dumps({'i': i, 'text': text, 'gpt2_tokens': len(ids), 'predicts_eot': not truncated,
                                  'bytes': len(raw)}, ensure_ascii=False) + '\n')
    if hashlib.file_digest(temp.open('rb'), 'sha256').hexdigest() != DOCUMENTS_SHA256:
        temp.unlink()
        raise ValueError('Rebuilt FineWeb documents differ from the pinned window')
    temp.replace(target)
    return target


def documents_path():
    """Prefer a verified mirror supplied by the runner; otherwise rebuild from Hugging Face."""
    sources = json.loads(os.environ.get('DELTA_DATASET_SOURCES', '{}'))
    if NAME in sources:
        path = Path(sources[NAME]['path']) / 'docs.jsonl'
    else:
        from filelock import FileLock
        cache = Path(os.environ.get('DELTA_DATASET_CACHE', Path(tempfile.gettempdir()) / 'delta-benchmark-datasets')) / NAME
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / 'docs.jsonl'
        with FileLock(str(path) + '.lock', timeout=-1):
            if not path.exists():
                build_documents(path)
    if hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != DOCUMENTS_SHA256:
        raise ValueError('FineWeb documents integrity')
    return path


def load_documents(bounds=None):
    with documents_path().open() as f:
        docs = [json.loads(line) for line in f]
    if len(docs) != DOCUMENTS:
        raise ValueError('FineWeb document count')
    return docs[bounds[0]:bounds[1]] if bounds else docs


def windows(doc, tokenizer, eot, size=SCORING_WINDOW):
    """Prompts whose scored targets cover the document (and its closing <eot>) exactly once."""
    targets = tokenizer(doc['text'], add_special_tokens=False)['input_ids'] + ([eot] if doc['predicts_eot'] else [])
    if not targets:
        return []
    step = size - 1
    return [[eot] + targets[i:i + step] for i in range(0, len(targets), step)]


def reduce_response(request, response):
    """Replace per-token prompt log-probabilities with their sum when the client asked for it.

    Native vLLM returns one entry per prompt token (about 100 bytes each). The sum is
    all this benchmark needs; reducing it keeps caches and stored responses small.
    """
    if request.get('delta_reduce') != REDUCE or not isinstance(response, dict):
        return response
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or 'delta_prompt_logprob_sum' in choices[0]:
        return response
    prompt = request.get('prompt')
    entries = choices[0].get('prompt_logprobs')
    if not isinstance(prompt, list) or not isinstance(entries, list) or len(entries) != len(prompt):
        return response
    total = 0.0
    for token, entry in zip(prompt[1:], entries[1:]):
        value = entry[str(token)]
        total += value['logprob'] if isinstance(value, dict) else value
    choice = {**choices[0], 'prompt_logprobs': None, 'delta_prompt_logprob_sum': total,
              'delta_prompt_logprob_count': len(prompt) - 1}
    return {**response, 'choices': [choice]}


def logprob_sum(prompt, response):
    """Sum of target log-probabilities from a reduced or native vLLM prompt_logprobs response."""
    choice = response['choices'][0]
    if 'delta_prompt_logprob_sum' in choice:
        total, count = choice['delta_prompt_logprob_sum'], choice['delta_prompt_logprob_count']
    else:
        entries = choice.get('prompt_logprobs')
        if not isinstance(entries, list) or len(entries) != len(prompt):
            raise ValueError('Endpoint did not return prompt log-probabilities')
        total, count = 0.0, 0
        for token, entry in zip(prompt[1:], entries[1:]):
            value = entry[str(token)]
            total += value['logprob'] if isinstance(value, dict) else value
            count += 1
    if count != len(prompt) - 1 or type(total) not in (int, float) or not math.isfinite(total) or total > 0:
        raise ValueError('Invalid prompt log-probability coverage')
    return float(total)


def evaluate(root, endpoint, config, phase_callback=None):
    """Score a document slice through the caching proxy; return the summary entry."""
    import httpx
    from transformers import AutoTokenizer
    root = Path(root)
    p = config['protocol']
    if p.get('documents_sha256') != DOCUMENTS_SHA256 or p.get('scoring_window_tokens') != SCORING_WINDOW:
        raise ValueError('FineWeb protocol identity')
    bounds = config.get('task_slice')
    started = time.monotonic()
    docs = load_documents(bounds)
    tokenizer = AutoTokenizer.from_pretrained(str(root / 'model'))
    eot = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if not isinstance(eot, int) or eot == tokenizer.unk_token_id:
        raise ValueError('Tokenizer has no <|endoftext|> token')
    jobs = [(d['i'], w) for d in docs for w in windows(d, tokenizer, eot)]
    url = endpoint.rstrip('/').removesuffix('/v1') + '/v1/completions'
    model = config.get('inference_model') or 'candidate'
    client = httpx.Client(timeout=httpx.Timeout(None, connect=60))

    def score(job):
        _, prompt = job
        body = {'model': model, 'prompt': prompt, 'max_tokens': 1, 'temperature': 0.0,
                'prompt_logprobs': 0, 'delta_reduce': REDUCE}
        response = client.post(url, json=body)
        response.raise_for_status()
        return logprob_sum(prompt, response.json())

    with ThreadPoolExecutor(max_workers=config.get('task_concurrency', p.get('concurrency', 32))) as pool:
        sums = list(pool.map(score, jobs))
    client.close()
    if phase_callback:
        phase_callback('scoring')
    per_doc = {d['i']: {'i': d['i'], 'nll': 0.0, 'targets': 0, 'windows': 0,
                        'gpt2_targets': d['gpt2_tokens'] + d['predicts_eot'], 'bytes': d['bytes']} for d in docs}
    for (i, prompt), total in zip(jobs, sums):
        row = per_doc[i]
        row['nll'] -= total
        row['targets'] += len(prompt) - 1
        row['windows'] += 1
    rows = list(per_doc.values())
    totals = {'nll_sum_nats': sum(r['nll'] for r in rows), 'candidate_targets': sum(r['targets'] for r in rows),
              'gpt2_targets': sum(r['gpt2_targets'] for r in rows), 'utf8_bytes': sum(r['bytes'] for r in rows),
              'windows': len(jobs), 'split_documents': sum(r['windows'] > 1 for r in rows)}
    metrics = {'primary_metric': 'loss_per_gpt2_token', 'lower_is_better': True, 'totals': totals, **derived(totals)}
    report = {'benchmark': NAME, 'protocol': {k: p.get(k) for k in protocol_fields()}, 'slice': bounds,
              'metrics': metrics, 'documents': rows}
    with gzip.open(root / (NAME + '.report.json.gz'), 'wt') as stream:
        json.dump(report, stream)
    return {NAME: {'completed_items': len(docs), 'expected_items': len(docs), 'metrics': metrics,
                   'performance': {'wallclock_seconds': time.monotonic() - started,
                                   'scored_tokens_per_second': totals['candidate_targets'] / max(time.monotonic() - started, 1e-3)}}}


def derived(totals):
    """Metrics computed from additive totals, so shards combine exactly."""
    nll = totals['nll_sum_nats']
    return {'value': nll / totals['gpt2_targets'], 'loss_per_gpt2_token': nll / totals['gpt2_targets'],
            'nats_per_token': nll / totals['candidate_targets'],
            'bits_per_byte': nll / math.log(2) / totals['utf8_bytes']}


def combine(parts):
    """Aggregate shard metrics into one result."""
    keys = ('nll_sum_nats', 'candidate_targets', 'gpt2_targets', 'utf8_bytes', 'windows', 'split_documents')
    totals = {k: sum(p['totals'][k] for p in parts) for k in keys}
    return {'primary_metric': 'loss_per_gpt2_token', 'lower_is_better': True, 'totals': totals, **derived(totals)}
