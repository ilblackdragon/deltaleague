"""Run the registered third overall comparison of the MLP channel graft."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from final_channel_panel import paired_broad
from routing_runtime import HERE, configure, metal_slot


def evaluate(root, protocol, trial):
    from evaluate_round import check_versions, native, native_text, previous, prior

    check_versions()
    first = native.read(root / 'records/decision.json')
    assert first['nominate'] is None and first['fresh'] is not None
    arm = protocol['candidate']
    assert first['arm'] == arm and trial['parent'] == protocol['parent']
    assert native.read(root / arm / 'build.json')['patch_sha256'] == protocol['patch_sha256']
    reference = root.parent / protocol['baseline_run']
    suite = prior.suite_from(reference / 'suite.json', protocol['suite_sha256'])
    ids = native.expected_ids(suite)
    assert len(ids) == 18 and sum(map(len, ids.values())) == protocol['items']
    for source, digest in ((root / 'fresh-suite.json', protocol['fresh_sha256']),
                           (root.parent / 'reasoning-route-generalization-v1/suite.json', trial['confirmation']['suite_sha256']),
                           (root.parent / 'leader-round2-v1/general/suite.json', trial['development']['suite_sha256'])):
        other = native.expected_ids(prior.suite_from(source, digest))
        assert all(ids[g].isdisjoint(items) for g, items in other.items())
    evaluate_tasks = native_text.evaluate_tasks

    def long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return evaluate_tasks(**kwargs)

    with patch.object(native_text, 'evaluate_tasks', long_requests):
        third = paired_broad(root, arm, suite, 'third-broad', protocol['seed'], trial['parent_hash'],
                             reference, protocol['baseline_alias'])
    second = native.read(root / 'records' / ('confirmation-' + arm + '.json'))
    cohorts = [second['by_benchmark'], first['fresh'], third['by_benchmark']]
    nets = [{g: rows[g]['wins'] - rows[g]['losses'] for g in previous.FOCUS} for rows in cohorts]
    pooled = {g: sum(c[g] for c in nets) for g in previous.FOCUS}
    broad = [first['development']['delta'], second['macro']['delta'], third['macro']['delta']]
    qualifies = min(broad) > 0 and min(pooled.values()) >= 0 and sum(pooled.values()) > 0
    result = dict(arm=arm, third=third, broad=broad, reasoning_by_cohort=nets, pooled=pooled,
                  strict_joint=min(pooled.values()) > 0, nominate=arm if qualifies else None)
    previous.record(root, 'overall-decision', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    trial = json.loads((HERE / 'mlp-channels/protocol.json').read_text())
    rule = json.loads((HERE / 'mlp-channels/overall-protocol.json').read_text())
    configure(trial)
    root = args.output.resolve()
    with metal_slot(root):
        evaluate(root, rule, trial)
