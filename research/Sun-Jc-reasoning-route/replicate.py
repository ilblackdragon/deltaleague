"""Evaluate frozen near-miss candidates on a second, previously unscored cohort."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from routing_runtime import HERE, configure

def run(output, original_root, union_root):
    protocol = json.loads((HERE / 'replication/protocol.json').read_text())
    configure(json.loads((HERE / 'protocol.json').read_text()))
    from evaluate_ablation import compare
    from evaluate_round import check_versions, native, native_text, previous, prior

    check_versions()
    (output / 'records').mkdir(parents=True, exist_ok=True)
    old_suite = prior.suite_from(original_root / 'fresh-suite.json', protocol['selection_sha256'])
    fresh = prior.suite_from(union_root / 'fresh-suite.json', protocol['replication_sha256'])
    for group in previous.FOCUS:
        assert native.expected_ids(old_suite)[group].isdisjoint(native.expected_ids(fresh)[group])
    union_arm = 'late-plus-recurrent'
    first = native.read(original_root / 'records/decision.json')
    assert first['nominate'] is None
    seed = protocol['seed']
    official_evaluate = native_text.evaluate_tasks

    def evaluate_long_requests(**kwargs):
        kwargs['model'].timeout = 1800
        return official_evaluate(**kwargs)

    with patch.object(native_text, 'evaluate_tasks', evaluate_long_requests):
        if not (union_root / 'records' / ('fresh-' + union_arm + '.json')).exists():
            previous.fresh_result(union_root, union_arm, fresh, seed, None)
        second = {}
        for arm in protocol['candidates']:
            target = output / arm
            target.mkdir(exist_ok=True)
            for filename in ('plan.json', 'build.json'):
                source = original_root / arm / filename
                dest = target / filename
                if not dest.exists():
                    dest.write_bytes(source.read_bytes())
                assert dest.read_bytes() == source.read_bytes()
            second[arm] = compare(output, arm, 'fresh', fresh, seed, union_root, union_arm)
    eligible = []
    evidence = {}
    for arm in protocol['candidates']:
        cohorts = [first['fresh'][arm], second[arm]]
        nets = [{g: v['wins'] - v['losses'] for g, v in cohort.items()} for cohort in cohorts]
        pooled = {g: sum(net[g] for net in nets) for g in previous.FOCUS}
        percentages = [100 * pooled[g] / sum(c[g]['items'] for c in cohorts) for g in previous.FOCUS]
        broad = [first[p][arm]['delta'] for p in ('development', 'confirmation')]
        gain = (min(broad) > 0 and all(min(n.values()) >= 0 for n in nets)
                and min(pooled.values()) > 0 and sum(nets[1].values()) > 0)
        evidence[arm] = dict(broad=broad, first=cohorts[0], second=cohorts[1], pooled=pooled, qualified=gain)
        if gain:
            eligible.append((arm, min(percentages), sum(broad)))
    qualified = [item[0] for item in sorted(eligible, key=lambda x: (x[1], x[2]), reverse=True)]
    previous.record(output, 'decision', dict(evidence=evidence, qualified=qualified,
                                            nominate=qualified[0] if qualified else None))
    native.note(output, 'complete', qualified[0] if qualified else 'none', 'all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for path in ('output', 'original', 'union'):
        parser.add_argument(path, type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.original.resolve(), args.union.resolve())
