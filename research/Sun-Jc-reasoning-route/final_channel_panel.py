"""Compare a frozen public-weight transplant with the official parent."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from routing_runtime import HERE, configure, metal_slot


def paired_broad(root, arm, confirm, phase, seed, parent_hash, reference, reference_arm):
    from evaluate_round import native, prior, proven_parent

    dest = root / phase
    dest.mkdir(exist_ok=True)
    alias = arm + '-' + phase
    plan = native.read(root / arm / 'plan.json')
    plan['candidates'] = {alias: plan['candidates'][arm]}
    plan['generation']['seed'] = seed
    work = root / alias
    work.mkdir(exist_ok=True)
    if (work / 'plan.json').exists():
        assert native.read(work / 'plan.json') == plan
    native.write(work / 'plan.json', plan)
    groups = native.order(confirm)
    child = {}
    with native.serve(root, alias, dest) as (_, health):
        development = native.read(root / arm / 'development/backend.json')
        assert health['identities'][alias] == development['identities'][arm]
        assert health['identities']['parent']['official_content_hash'] == parent_hash
        parent, proof = proven_parent(reference / 'paired', reference / reference_arm / 'plan.json',
                                      confirm, health, plan['generation'], groups)
        native.write(dest / 'parent-reuse.json', dict(source=str(reference), reports=proof))
        for group in groups:
            native.note(root, phase, arm, group)
            child.update(native.run_group(confirm, group, alias, plan, dest))
    report = native.full_report(dest, confirm, parent, child, arm)
    macro = prior.summarize(report, arm)
    return dict(macro=macro, by_benchmark=native.compare_groups(parent, child, native.expected_ids(confirm)),
                suite_sha256=confirm['sha256'], seed=seed,
                patch_sha256=native.read(root / arm / 'build.json')['patch_sha256'])


def run(root, rule):
    from evaluate_ablation import compare
    from evaluate_round import check_versions, native, native_text, previous, prior

    check_versions()
    arm, = rule['arms']
    base = root.parent
    reference = base / 'reasoning-route-generalization-v1'
    dev = prior.suite_from(base / 'leader-round2-v1/general/suite.json', rule['development']['suite_sha256'])
    confirm = prior.suite_from(reference / 'suite.json', rule['confirmation']['suite_sha256'])
    fresh_receipt = native.read(root / 'fresh-selection.json')
    if 'holdout' in rule:
        assert fresh_receipt['suite_sha256'] == rule['holdout']['sha256']
    fresh = prior.suite_from(root / 'fresh-suite.json', fresh_receipt['suite_sha256'])
    dev_ids, confirm_ids, fresh_ids = map(native.expected_ids, (dev, confirm, fresh))
    assert len(dev_ids) == len(confirm_ids) == 18 and sum(map(len, confirm_ids.values())) == 332
    assert all(dev_ids[g].isdisjoint(confirm_ids[g]) for g in dev_ids)
    assert all(fresh_ids[g].isdisjoint(dev_ids[g] | confirm_ids[g]) for g in previous.FOCUS)
    assert 'Manifest and artifact hashes are valid' in (root / arm / 'validation.log').read_text()
    (root / 'records').mkdir(exist_ok=True)
    original = native_text.evaluate_tasks

    def long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return original(**kwargs)

    def confirmation():
        comparison = paired_broad(root, arm, confirm, 'confirmation', rule['confirmation']['seed'],
                                  rule['parent_hash'], reference, 'late-block')
        previous.record(root, 'confirmation-' + arm, comparison)
        return comparison['macro']

    with patch.object(native_text, 'evaluate_tasks', long_requests):
        development = compare(root, arm, 'development', dev, rule['development']['seed'],
                              base / 'leaderd012-transfer-v1', 'attention-transfer')
        second = confirmation() if development['delta'] > 0 else None
        reasoning = None
        if second is not None and second['delta'] > 0:
            reasoning = previous.fresh_result(root, arm, fresh, rule['fresh']['seed'], None)
        qualifies = reasoning is not None and all(reasoning[g]['wins'] > reasoning[g]['losses']
                                                for g in previous.FOCUS)
        previous.record(root, 'decision', dict(arm=arm, development=development, confirmation=second,
                                               fresh=reasoning, nominate=arm if qualifies else None))
        native.note(root, 'complete', arm, 'all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--protocol', type=Path, default=HERE / 'final-channel/protocol.json')
    args = parser.parse_args()
    rule = json.loads(args.protocol.read_text())
    configure(rule)
    root = args.output.resolve()
    with metal_slot(root):
        run(root, rule)
