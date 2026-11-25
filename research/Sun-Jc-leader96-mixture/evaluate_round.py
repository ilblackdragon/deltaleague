"""Compare registered edits with newly generated evidence from the official leader."""
import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

from delta_cli.public_eval import text as native_text
from delta_cli.public_eval.common import check_versions

LAB = Path(os.environ.get('DELTA_RETENTION_LAB',
    '/Users/jc/delta/experiments/recurrent-retention/research/Sun-Jc-recurrent-retention'))
if hashlib.sha256((LAB/'run.py').read_bytes()).hexdigest() != '706705c2a8dbc5cc29bc25816e46d6fa5ee15ff26b26bedb5b7c22e7c5c3b955':
    raise RuntimeError('Unexpected version of the paired evaluation runner')
sys.path.insert(0, str(LAB))
import run as previous

native = previous.native
prior = previous.prior


def proven_parent(source, plan, suite, health, generation, groups):
    old = native.read(source/'backend.json')
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == old['plan_sha256']
    assert native.read(plan)['generation'] == generation
    assert old['identities']['parent'] == health['identities']['parent']
    for field in ('runtime','execution','dtype','generation_seed','server_sha256','adapter_sha256'):
        assert old[field] == health[field]
    parent = {}
    reports = {}
    for group in groups:
        path = source/'parent'/(group+'.json')
        reports[group] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert native.read(path.with_suffix('.verified.json')) == {
            'report_sha256':reports[group],'suite_sha256':suite['sha256']}
        parent.update(native.rows_from_report(suite,group,path))
    ids = {group: values for group, values in native.expected_ids(suite).items() if values}
    assert native.indexed(parent,ids) == native.indexed(native.read(source/'groups.json')['parent'],ids)
    return parent, reports


def develop(root, arm, reference_arm, suite, rule):
    dest = root/arm/'development'
    dest.mkdir(exist_ok=True)
    assert 'Manifest and artifact hashes are valid' in (root/arm/'validation.log').read_text()
    parent, child = {}, {}
    with native.serve(root,arm,dest) as (plan,health):
        assert health['identities']['parent']['official_content_hash'] == rule['content_hash']
        assert health['generation_seed'] == rule['development']['seed']
        if arm != reference_arm:
            reference = root / reference_arm
            parent, proof = proven_parent(reference / 'development', reference / 'plan.json', suite, health,
                                          plan['generation'], native.order(suite))
            native.write(dest/'parent-reuse.json',dict(source=reference_arm,reports=proof))
        for i,group in enumerate(native.order(suite)):
            if arm==reference_arm:
                models = ('parent',arm) if i%2==0 else (arm,'parent')
            else:
                models = (arm,)
            for key in models:
                native.note(root,'development',key,group)
                target = parent if key=='parent' else child
                target.update(native.run_group(suite,group,key,plan,dest))
    native.full_report(dest,suite,parent,child,arm)
    return previous.archive_panel(root,dest,'development',arm,rule['development']['seed'])


def evaluate(root, rule):
    check_versions()
    (root/'records').mkdir(exist_ok=True)
    dev = prior.suite_from(root.parent/'leader-round2-v1/general/suite.json',rule['development']['suite_sha256'])
    conf = prior.suite_from(root.parent/'ten-hour-confirmation-v1/suite.json',rule['confirmation']['suite_sha256'])
    selection = native.read(root/'fresh-selection.json')
    fresh = prior.suite_from(root/'fresh-suite.json',selection['suite_sha256'])
    dev_ids, conf_ids, fresh_ids = map(native.expected_ids,(dev,conf,fresh))
    assert len(dev_ids)==len(conf_ids)==18 and all(dev_ids[g].isdisjoint(conf_ids[g]) for g in dev_ids)
    assert sum(map(len,dev_ids.values()))==sum(map(len,conf_ids.values()))==204
    assert all(fresh_ids[g].isdisjoint(dev_ids[g] | conf_ids[g]) for g in previous.FOCUS)
    previous.record(root,'fresh-selection',selection)
    results={}
    reference_arm=next(iter(rule['arms']))
    for arm in rule['arms']:
        results[arm]=develop(root,arm,reference_arm,dev,rule)
    best=max(results,key=lambda name:previous.rank(results[name]))
    patch_hash=native.read(root/best/'build.json')['patch_sha256']
    previous.record(root,'selection',dict(arm=best,patch_sha256=patch_hash,ranking_rule=rule['selection']))
    seq='confirmation'
    groups=prior.compare(root,best,conf,seq,rule[seq]['seed'],native.order(conf))
    native.full_report(root/seq,conf,groups['parent'],groups[best],best)
    confirmed=previous.archive_panel(root,root/seq,seq,best,rule[seq]['seed'])
    panel=previous.fresh_result(root,best,fresh,rule['fresh']['seed'],None)
    broad=min(results[best]['delta'],confirmed['delta'])>0
    joint=all(panel[g]['candidate_correct']>panel[g]['parent_correct'] for g in previous.FOCUS)
    previous.record(root,'decision',dict(arm=best,patch_sha256=patch_hash,development=results[best],
                    confirmation=confirmed,fresh=panel,broad_replication=broad,joint_fresh_gain=joint,
                    full_objective=broad and joint))
    native.note(root,'complete',best,'all')


if __name__=='__main__':
    original=native_text.evaluate_tasks

    def evaluate_long_requests(**kwargs):
        kwargs['model'].timeout=1800
        return original(**kwargs)

    with patch.object(native_text,'evaluate_tasks',evaluate_long_requests):
        protocol_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('protocol.json')
        evaluate(Path(sys.argv[1]).resolve(), native.read(protocol_path))
