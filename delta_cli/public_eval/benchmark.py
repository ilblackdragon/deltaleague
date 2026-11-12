"""Standalone full-set benchmark worker; scoped object capabilities, no DB credentials."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def final_answer(text, thinking):
    if not thinking:
        return text
    # The rendered prompt already opens <think>. An unfinished thought is not an answer.
    return text.rsplit('</think>', 1)[1].strip() if '</think>' in text else ''


def read_native_evidence(path):
    """BFCL uses .json for both JSON documents and JSON Lines checker output."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Parse every line strictly; malformed evidence must still fail publication.
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def evaluate(root, phase_callback=None):
    from .history import History, ResponseProxy, export_checkpoint, import_checkpoint
    config = json.loads((root/'manifest.json').read_text())
    history = History(root/'history.sqlite3',seed_path=config.get('history_seed_path'))
    with history.connect() as db: db.execute('PRAGMA journal_mode=WAL')
    if not config.get('history_seed_path'):import_checkpoint(history, 'public', root/'history-seed.sqlite3.gz')
    counts = dict(requests=0, prompt_tokens=0, completion_tokens=0, truncated=0, unfinished_thinking=0)
    lock = threading.Lock()
    config = json.loads((root/'manifest.json').read_text())
    p = config['protocol']
    group = p['benchmarks'][0]
    started = time.monotonic()
    raw = gzip.open(root/(group+'.raw.jsonl.gz'), 'wt')
    def observe(request, response):
        with lock:
            raw.write(json.dumps({'request':request, 'response':response})+'\n')
            counts['requests'] += 1
            for key in ('prompt_tokens', 'completion_tokens'):
                counts[key] += response.get('usage',{}).get(key,0)
            for choice in response['choices']:
                # Likelihood requests generate one token by design; that is not truncation.
                counts['truncated'] += choice.get('finish_reason') == 'length' and not request.get('delta_reduce')
                text = choice.get('text', choice.get('message',{}).get('content')) or ''
                counts['unfinished_thinking'] += request.get('max_tokens',0) > 1 and p['thinking'] and '</think>' not in text
            status = {'stage':'evaluating','benchmark':group, 'requests_completed':counts['requests'],
                      'benchmark_total':p['expected_items'], 'benchmark_seconds':time.monotonic()-started,
                      'tokens':counts['completion_tokens']}
            # Requests are not questions for likelihood or multi-turn benchmarks.
            temp=root/'progress.tmp';temp.write_text(json.dumps(status));temp.replace(root/'progress.json')
    with ResponseProxy(history, 'public', upstream=config.get('inference_upstream','http://127.0.0.1:8000'), upstream_model=config.get('inference_model'), api_key_env=config.get('api_key_env'), request_timeout=None, observer=observe) as proxy:
        stop = threading.Event()
        def snapshot():
            while not stop.wait(300):
                try: export_checkpoint(history, 'public', root/'history.sqlite3.gz')
                except Exception: traceback.print_exc()
        thread = threading.Thread(target=snapshot, daemon=True);thread.start()
        try:
            summary = evaluate_with_proxy(root, proxy, phase_callback=phase_callback)
            summary[group]['performance'].update(counts, reused_requests=proxy.hits, new_requests=proxy.misses)
            temp=root/'summary.tmp';temp.write_text(json.dumps(summary,default=str));temp.replace(root/'summary.json')
            return summary
        finally:
            stop.set();thread.join();raw.close()
            export_checkpoint(history, 'public', root/'history.sqlite3.gz')


def evaluate_with_proxy(root, proxy, phase_callback=None):
    from . import fineweb_loss
    config = json.loads((root/'manifest.json').read_text());p=config['protocol']
    if p['benchmarks'][0] == fineweb_loss.NAME:
        return fineweb_loss.evaluate(root, proxy.endpoint, config, phase_callback=phase_callback)
    from contextlib import ExitStack
    from unittest.mock import patch
    import httpx
    from delta_cli.public_eval import text, coding, tools
    from delta_cli.public_eval.common import check_versions
    from delta_cli.public_eval.suite import manifest, PRIMARY
    from .coverage import expand, populations
    import lm_eval.models.openai_completions as api
    check_versions()
    suite = manifest()
    assert suite['sha256'] == p['suite_sha256']
    group = p['benchmarks'][0]
    started = time.monotonic()
    expanded = expand(suite, group) if p.get('coverage', 'full') == 'full' or p.get('selection_basis') == 'full' else suite
    if group.startswith('bfcl_'):
        from .coverage import categories
        expanded['bfcl'] = {c: expanded['bfcl'][c] for c in categories(expanded, group)}
    expanded['seed'] = p['seed']
    expected = (populations(suite)[group] if p.get('coverage', 'full') == 'full' or p.get('selection_basis') == 'full'
                else sum(e['count'] for e in expanded['bfcl'].values()) if group.startswith('bfcl_')
                else expanded['text' if group in expanded['text'] else 'code'][group]['count'])
    if config.get('task_slice') is not None:
        from .coverage import slice_suite
        expanded=slice_suite(expanded,group,config['task_slice'])
        expected=config['task_slice'][1]-config['task_slice'][0]
    assert expected == config.get('task_count',p['expected_items'])
    output = root/(group+'.report.json')
    generation = p.get('generation',dict(p['sampling'], enable_thinking=p['thinking'],max_tokens=p['max_tokens']))
    Original = api.LocalCompletionsAPI
    class LongRunningAPI(Original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout = None
    post = httpx.post
    def long_post(url, *args, **kwargs):
        if str(url).startswith(proxy.endpoint): kwargs['timeout'] = httpx.Timeout(None, connect=60)
        return post(url,*args,**kwargs)
    handler_for = tools.handler_for
    def long_handler(*args, **kwargs):
        handler, alias = handler_for(*args,**kwargs)
        create = handler.client.completions.create
        def long_create(*args,**kwargs):
            kwargs['timeout'] = None
            return create(*args,**kwargs)
        handler.client.completions.create = long_create
        return handler,alias
    with ExitStack() as stack:
        stack.enter_context(patch.object(api,'LocalCompletionsAPI',LongRunningAPI))
        if config.get('task_slice') is not None and group in suite['text']:
            original_load=text.load_tasks
            selection=expanded['text'][group]['selection']
            def shard_load(*args,**kwargs):
                return {n:t for n,t in original_load(*args,**kwargs).items() if n in selection}
            stack.enter_context(patch.object(text,'load_tasks',shard_load))
        stack.enter_context(patch.object(httpx,'post',long_post))
        stack.enter_context(patch.object(tools,'handler_for',long_handler))
        if group in suite['text']:
            result = text.run(expanded,group,proxy.endpoint,'candidate',str(root/'model'),output,
                              concurrency=config.get('task_concurrency',p['concurrency']),generation=generation)
            metric, selected_filter = PRIMARY[group]
            values = [float(r[metric]) for rows in result['samples'].values() for r in rows if r.get('filter') == selected_filter]
            if len(values) != expected: raise ValueError('Full metric coverage mismatch')
            metrics = {'primary_metric':metric,'score_percent':100*sum(values)/expected,
                       'tasks':result['results'],'groups':result.get('groups',{})}
        elif group in suite['code']:
            result = coding.run(expanded,group,proxy.endpoint,'candidate','',output,sandbox_backend='wasi',
                                concurrency=config.get('task_concurrency',p['concurrency']),generation=generation)
            if result['count'] != expected: raise ValueError('Incomplete coding coverage')
            metrics = {'primary_metric':'pass@1','score_percent':100*result['score']}
        else:
            result = tools.run(expanded,proxy.endpoint,'candidate',str(root/'model'),output,
                               concurrency=config.get('task_concurrency',p['concurrency']),generation=generation,phase_callback=phase_callback)
            total = sum(v['count'] for v in result['scores'].values())
            if total != expected: raise ValueError('Incomplete BFCL coverage')
            metrics = {'primary_metric':'accuracy','score_percent':100*sum(v['accuracy']*v['count'] for v in result['scores'].values())/total,
                       'categories':result['scores']}
            # Include native checker evidence as well as the category summary.
            result['scope'] = 'Full covered public BFCL tracks; excludes web search and memory'
            result['evidence'] = {str(f.relative_to(output.with_suffix(''))):read_native_evidence(f)
                                  for f in output.with_suffix('').rglob('*.json')}
    if phase_callback: phase_callback('scoring')
    with gzip.open(root/(group+'.report.json.gz'),'wt') as stream: json.dump(result,stream,default=str)
    return {group:{'completed_items':expected,'expected_items':expected,'metrics':metrics,
                   'performance':{'wallclock_seconds':time.monotonic()-started}}}

