"""Run user-supplied text, Python and Harbor tasks through the public adapters.

No dataset, checkpoint scheduling, credentials or publication policy lives here.
The caller must be trusted with reference answers; candidate code stays sandboxed.
"""
import concurrent.futures
import json
import time
from pathlib import Path

from .custom_tasks import chat_request, answer, coding_prompt, candidate_files, check_python


def evaluate_item(task, *, endpoint, generation, seed, root, tokenizer, harbor_root,
                  image, epoch=0, inference_finished=None):
    import httpx
    started = time.monotonic()
    row = {'id': task['id'], 'track': task['track'], 'state': 'complete'}
    evidence = {}
    if task['track'] == 'terminal':
        from .custom_tasks import bounded_agent_endpoint
        from .terminal import trial_config, trial_name, valid_reward, task_evidence
        from .native_trial import run_trial_isolated
        limits = task['limits']
        protocol = {'generation': generation, 'seed': seed,
                    'max_model_len': generation['max_model_len'],
                    'agent_options': {'parser_name': 'json', 'enable_summarize': True,
                        'proactive_summarization_threshold': 8000, 'max_turns': limits['max_turns']}}
        name = trial_name(root, task['id'], 0, epoch)
        trials = root / 'trials'
        budget_stop=trials/(name+'.budget-stop')
        trials.mkdir(parents=True,exist_ok=True)
        with bounded_agent_endpoint(endpoint, protocol, 0,
                output_limit=limits['max_total_output_tokens'], call_limit=limits['max_turns'], budget_stop_path=budget_stop) as (agent_endpoint, stats):
            cfg = trial_config(harbor_root / task['split'] / task['id'], trials, name, agent_endpoint, protocol)
            native = run_trial_isolated(cfg, tokenizer_path=tokenizer, budget_stop_path=budget_stop, inference_finished=inference_finished)
        # Even budget exhaustion requires a real verifier result: infrastructure
        # failures must not silently become incorrect model answers.
        reward = valid_reward(native)
        check = json.loads((trials / name / 'verifier/check.json').read_text())
        if check.get('task') != task['id'] or check.get('passed') != bool(reward):
            raise ValueError('Verifier identity/reward mismatch')
        row.update(stats)
        row.update(passed=bool(reward) and not stats['budget_exhausted'],
                   reason='agent_output_budget' if stats['budget_exhausted'] else ('passed' if reward else 'task_failure'))
        evidence = {'native': native.model_dump(mode='json'), 'trial': task_evidence(trials / name)}
    else:
        prompt = coding_prompt(task['prompt'], task['starter_files']) if task['track'] == 'coding' else task['prompt']
        response = httpx.post(endpoint, json=chat_request(prompt, generation, seed), timeout=None)
        response.raise_for_status()
        data = response.json()
        if inference_finished:inference_finished()
        from .history import valid_response
        if not valid_response(data):
            raise ValueError('Incomplete model response')
        text = answer(data, generation['enable_thinking'])
        row.update(data.get('usage', {}))
        row['inference_seconds'] = time.monotonic() - started
        evidence = {'response': data}
        if task['track'] == 'coding':
            try:
                files = {**task['starter_files'], **candidate_files(text, task['starter_files'], entrypoint=task.get('entrypoint'))}
            except ValueError:
                row.update(passed=False, reason='invalid_code_response')
            else:
                row.update(check_python(files, [c['input'] for c in task['hidden_cases']],
                    [c['expected'] for c in task['hidden_cases']], image=image, entrypoint=task.get('entrypoint')))
        else:
            from delta_cli.scoring import score_answer
            row.update(passed=bool(score_answer(task, text)), reason='exact_answer')
    row['wallclock_seconds'] = time.monotonic() - started
    return row, evidence


def evaluate(tasks, *, concurrency=8, progress=None, inference_finished=None, **kwargs):
    tasks = list(tasks)
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Empty or duplicate custom tasks')
    import threading
    lock=threading.Lock();finished=set()
    def finished_item(key):
        with lock:
            finished.add(key)
            if len(finished)==len(tasks) and inference_finished:inference_finished()
    rows, evidence = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(evaluate_item, t, inference_finished=lambda key=t['id']:finished_item(key), **kwargs): t['id'] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            row, raw = future.result()
            rows[row['id']] = row
            evidence[row['id']] = raw
            if progress:
                progress(len(rows), len(tasks))
    return [rows[t['id']] for t in tasks], evidence
