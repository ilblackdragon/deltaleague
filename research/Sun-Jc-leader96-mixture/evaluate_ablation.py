"""Compare a locked module ablation with the same verified d0120ae6 responses."""
import sys
from pathlib import Path
from unittest.mock import patch

from evaluate_round import check_versions, native, native_text, previous, prior, proven_parent


def compare(root, arm, phase, suite, seed, parent_root, parent_arm):
    suffix = {'development': '', 'confirmation': '-confirmation', 'fresh': '-new-questions'}[phase]
    alias = arm + suffix
    reference_alias = parent_arm + suffix
    relative_path = {'development': Path(arm) / phase, 'confirmation': Path(phase), 'fresh': Path('fresh-' + arm)}[phase]
    dest = root / relative_path
    source = parent_root / str(relative_path).replace(arm, parent_arm)
    dest.mkdir(parents=True, exist_ok=True)
    plan = native.read(root / arm / 'plan.json')
    plan['generation']['seed'] = seed
    plan['candidates'] = {alias: plan['candidates'][arm]}
    work = root / alias
    work.mkdir(exist_ok=True)
    if (work / 'plan.json').exists():
        assert native.read(work / 'plan.json') == plan
    else:
        native.write(work / 'plan.json', plan)
    groups = previous.FOCUS if phase == 'fresh' else native.order(suite)
    child = {}
    with native.serve(root, alias, dest) as (_, health):
        assert health['identities']['parent']['official_content_hash'] == plan['parent_content_hash']
        assert health['generation_seed'] == seed
        parent, proof = proven_parent(source, parent_root / reference_alias / 'plan.json', suite, health, plan['generation'], groups)
        native.write(dest / 'parent-reuse.json', dict(source=str(source), reports=proof))
        for group in groups:
            native.note(root, phase, arm, group)
            child.update(native.run_group(suite, group, alias, plan, dest))
    if phase == 'fresh':
        for rows in (parent, child):
            native.indexed(rows, {g: native.expected_ids(suite)[g] for g in groups})
        native.write(dest / 'groups.json', {'parent': parent, arm: child})
        scores = native.compare_groups(parent, child, groups)
        previous.record(root, 'fresh-' + arm, dict(seed=seed, suite_sha256=suite['sha256'], scores=scores,
                                                   baseline_source=str(source)))
        return scores
    native.full_report(dest, suite, parent, child, arm)
    return previous.archive_panel(root, dest, phase, arm, seed)


def run(root):
    check_versions()
    rule = native.read(Path(__file__).resolve().parent / 'ablation/protocol.json')
    arm = next(iter(rule['arms']))
    assert 'Manifest and artifact hashes are valid' in (root / arm / 'validation.log').read_text()
    parent_root = root.parent / rule['parent_run']
    assert native.read(parent_root / 'fresh-selection.json')['suite_sha256'] == rule['fresh_suite_sha256']
    (root / 'records').mkdir(exist_ok=True)
    suites = {
        'development': prior.suite_from(root.parent / 'leader-round2-v1/general/suite.json', rule['development']['suite_sha256']),
        'confirmation': prior.suite_from(root.parent / 'ten-hour-confirmation-v1/suite.json', rule['confirmation']['suite_sha256']),
        'fresh': prior.suite_from(parent_root / 'fresh-suite.json', rule['fresh_suite_sha256']),
    }
    results = {phase: compare(root, arm, phase, suite, rule[phase]['seed'], parent_root, rule['parent_arm'])
               for phase, suite in suites.items()}
    broad = min(results[phase]['delta'] for phase in ('development', 'confirmation')) > 0
    joint = all(s['candidate_correct'] > s['parent_correct'] for s in results['fresh'].values())
    previous.record(root, 'decision', dict(arm=arm, patch_sha256=native.read(root / arm / 'build.json')['patch_sha256'],
                                          **results, broad_replication=broad, joint_fresh_gain=joint, full_objective=broad and joint))
    native.note(root, 'complete', arm, 'all')


if __name__ == '__main__':
    original = native_text.evaluate_tasks

    def evaluate_long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return original(**kwargs)

    with patch.object(native_text, 'evaluate_tasks', evaluate_long_requests):
        run(Path(sys.argv[1]).resolve())
