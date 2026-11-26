"""Freeze unseen benchmark items and score one fixed overall-ranking candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from routing_runtime import HERE, configure


def prepare(root, rule):
    from delta_cli.public_eval.common import digest
    from delta_cli.public_eval.suite import manifest
    from delta_cli.public_eval.text import load_tasks, verify_tasks
    from fresh_panel import PREVIOUS, freeze, question_key
    from evaluate_round import native

    official = manifest()
    broad = [native.read(root.parent / n) for n in ('leader-round2-v1/general/suite.json', 'ten-hour-confirmation-v1/suite.json')]
    research = native.read(HERE / 'residual/protocol.json')
    histories = list(broad)
    selections = dict(PREVIOUS, **research['extra_exclusions'])
    selections['reasoning-route-residual-v1/fresh-suite.json'] = native.read(HERE / 'replication/protocol.json')['replication_sha256']
    selections.update(rule.get('extra_exclusions', {}))
    for name, sha in selections.items():
        item = native.read(root.parent / name)
        assert item['sha256'] == sha == digest({k: v for k, v in item.items() if k != 'sha256'})
        histories.append(item)

    def pick(key, available, count):
        salt = rule['salt']
        values = sorted(available, key=lambda i: hashlib.sha256(f'{salt}/{key}/{i}'.encode()).digest())[:count]
        assert len(values) == count
        return sorted(values)

    suite = copy.deepcopy(official)
    suite.update(id=rule.get('suite_id', 'late-block-third-general-panel'), scope='local-independent-ranking-panel')
    for group, entry in suite['text'].items():
        tasks = load_tasks(entry['task'], suite['dataset_revisions'])
        for task, spec in entry['selection'].items():
            old = [h['text'][group]['selection'][task] for h in histories if group in h['text'] and task in h['text'][group]['selection']]
            used = {i for s in old for i in s['indices']}
            fingerprints = {h for s in old for h in s['doc_sha256']}
            docs = tasks[task].eval_docs
            if group in rule.get('outside_manifest', ('bbh', 'mmlu')):
                available = set(range(len(docs))) - used - set(spec['indices'])
            else:
                available = set(spec['indices']) - used
            if group in ('math500', 'bbh'):
                seen = {question_key(docs[i], group) for i in used}
                available = {i for i in available if question_key(docs[i], group) not in seen}
            count = rule['by_subtask'].get(group, rule.get(group, rule['text_default']))
            # These immutable sources validate both IDs and document content before scoring.
            chosen = pick(task, available, count)
            hashes = [digest(docs[i]) for i in chosen]
            assert not fingerprints.intersection(hashes)
            spec.update(indices=chosen, doc_sha256=hashes)
        entry['count'] = sum(len(s['indices']) for s in entry['selection'].values())
        verify_tasks(tasks, entry['selection'])
    for group, entry in suite['code'].items():
        previous_ids = {str(doc['task_id']) for h in histories if group in h['code'] for doc in h['code'][group]['docs']}
        docs = dict(zip(entry['indices'], entry['docs'], strict=True))
        available = {i for i, doc in docs.items() if str(doc['task_id']) not in previous_ids}
        chosen = pick(group, available, rule['code_per_group'])
        entry.update(indices=chosen, docs=[docs[i] for i in chosen], count=len(chosen))
    for group, entry in suite['bfcl'].items():
        previous = {i for h in histories if group in h['bfcl'] for i in h['bfcl'][group]['ids']}
        docs = dict(zip(entry['ids'], entry['doc_sha256'], strict=True))
        chosen = pick(group, set(docs) - previous, rule['bfcl_per_category'])
        entry.update(ids=chosen, doc_sha256=[docs[i] for i in chosen], count=len(chosen))
    suite.pop('sha256')
    suite['sha256'] = digest(suite)
    ids = native.expected_ids(suite)
    assert len(ids) == 18
    for h in histories:
        old = native.expected_ids(h)
        for g, other in old.items():
            assert ids[g].isdisjoint(other), (h['id'], g, ids[g] & other)
    freeze(root / 'suite.json', suite)
    freeze(root / 'selection.json', dict(sha256=suite['sha256'], items={g: len(v) for g, v in ids.items()},
                                         salt=rule['salt'], excluded=sorted({h['sha256'] for h in histories})))
    print(suite['sha256'], sum(map(len, ids.values())), flush=True)


def evaluate(root, rule):
    from evaluate_round import check_versions, native, native_text, previous, prior

    check_versions()
    selection = native.read(root / 'selection.json')
    suite = prior.suite_from(root / 'suite.json', selection['sha256'])
    first_root = root.parent / 'reasoning-route-v1'
    replication = native.read(root.parent / 'reasoning-route-replication-v1/records/decision.json')
    assert replication['nominate'] is None
    arm = rule['arm']
    src = first_root / arm
    assert native.read(src / 'build.json')['patch_sha256'] == rule['patch_sha256']
    dest = root / 'paired'
    dest.mkdir(exist_ok=True)
    arm_path = root / arm
    arm_path.mkdir(exist_ok=True)
    plan = native.read(src / 'plan.json')
    plan['generation']['seed'] = rule['seed']
    if (arm_path / 'plan.json').exists():
        assert native.read(arm_path / 'plan.json') == plan
    native.write(arm_path / 'plan.json', plan)
    original = native_text.evaluate_tasks

    def long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return original(**kwargs)

    parent, child = {}, {}
    with patch.object(native_text, 'evaluate_tasks', long_requests), native.serve(root, arm, dest) as (_, health):
        previous_health = native.read(src / 'development/backend.json')
        for key in ('runtime', 'execution', 'dtype', 'server_sha256', 'adapter_sha256', 'identities'):
            assert health[key] == previous_health[key]
        assert health['generation_seed'] == rule['seed']
        for index, group in enumerate(native.order(suite)):
            models = ('parent', arm) if index % 2 == 0 else (arm, 'parent')
            for model in models:
                native.note(root, 'generalization', model, group)
                rows = native.run_group(suite, group, model, plan, dest)
                (parent if model == 'parent' else child).update(rows)
    report = native.full_report(dest, suite, parent, child, arm)
    macro = prior.summarize(report, arm)
    rows = native.compare_groups(parent, child, native.expected_ids(suite))
    prior_data = replication['evidence'][arm]
    pooled = {g: prior_data['pooled'][g] + rows[g]['wins'] - rows[g]['losses'] for g in previous.FOCUS}
    broad = [*prior_data['broad'], macro['delta']]
    suitable = min(broad) > 0 and min(pooled.values()) >= 0 and sum(pooled.values()) > 0
    native.write(root / 'decision.json', dict(arm=arm, patch_sha256=rule['patch_sha256'],
                 suite_sha256=suite['sha256'], total=sum(selection['items'].values()), macro=macro,
                 broad=broad, by_benchmark=rows, pooled_reasoning=pooled, strict_joint=min(pooled.values()) > 0,
                 nominate=arm if suitable else None))
    native.note(root, 'complete', arm, 'all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--protocol', type=Path, default=HERE / 'generalization/protocol.json')
    args = parser.parse_args()
    configure(json.loads((HERE / 'protocol.json').read_text()))
    protocol = json.loads(args.protocol.read_text())
    (prepare if args.action == 'prepare' else evaluate)(args.root.resolve(), protocol)
