"""Freeze a new question panel disjoint from the known public evaluation manifests."""
import copy
import hashlib
import json
import re
from pathlib import Path
import sys

from delta_cli.public_eval.common import digest
from delta_cli.public_eval.suite import manifest
from delta_cli.public_eval.text import load_tasks, verify_tasks

PREVIOUS = {
    'recurrent-retention-v1/fresh-suite.json': '93c1f11daca4c6ff4e40fe86c5e555b34529078bca90bf58d3612d303f2c27dd',
    'recurrent-write-gate-v1/fresh-suite.json': '8a6106f0a190aa2682b31ee8b6eff81b31d45d93821fae50bc7ae847a0e0af5d',
    'recurrent-write-gate-v1/transfer-suite.json': '4e479fb7c63698d2eeb05f3f22902a957bc4e670ccb05e64efddea87b7ef47f6',
    'leader-round2-v1/general/suite.json': 'b22f0530da84f9f92d11cae15883270fc3e54d6c1f080e1db2f210ca19c73e00',
    'ten-hour-confirmation-v1/suite.json': 'ba6736dce10369669cc828d8306f8a2b2fd521f13a476115094156ea98d149d2',
    'reasoning-balance-v1/fresh-suite.json': '5ba7ada7beff690ce8774f90bcc9fb0523df7d525cee68f146ec2928b630b9a5',
}


def question_key(doc, group):
    field = 'problem' if group == 'math500' else 'input'
    return re.sub(r'\s+', ' ', doc[field]).strip().casefold()


def freeze(path, obj):
    if path.exists() and json.loads(path.read_text()) != obj:
        raise ValueError(f'Existing selection differs at {path}')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def prepare(root, protocol):
    rule = protocol['fresh']
    official = manifest()
    history = [official]
    for name, expected in (PREVIOUS | protocol.get('extra_exclusions', {})).items():
        old = json.loads((root.parent/name).read_text())
        assert old['sha256'] == expected == digest({k:v for k,v in old.items() if k != 'sha256'})
        history.append(old)
    suite = copy.deepcopy(official)
    suite.update(id=protocol.get('suite_id', 'leader96-mixture-new-reasoning'), scope='out-of-manifest-reasoning', code={}, bfcl={},
                 text={k:suite['text'][k] for k in ('math500','bbh')})
    for group, setup in suite['text'].items():
        tasks = load_tasks(setup['task'], suite['dataset_revisions'])
        occupied_text = set()
        occupied_hash = set()
        omitted = {}
        for old in history:
            selection = old['text'][group]['selection']
            verify_tasks(tasks, selection)
            for name, spec in selection.items():
                omitted.setdefault(name, set()).update(spec['indices'])
                occupied_hash.update(spec['doc_sha256'])
                occupied_text.update(question_key(tasks[name].eval_docs[i], group) for i in spec['indices'])
        for name, spec in setup['selection'].items():
            docs = tasks[name].eval_docs
            n = rule['math500'] if group=='math500' else rule['bbh']//len(setup['selection'])
            allowed = set(range(len(docs))) - omitted[name]
            ranked = sorted(allowed, key=lambda i: hashlib.sha256(
                f"{rule['salt']}:{name}:{i}".encode()).digest())
            chosen = []
            for index in ranked:
                doc = docs[index]
                text_key, fingerprint = question_key(doc, group), digest(doc)
                if text_key in occupied_text or fingerprint in occupied_hash:
                    continue
                chosen.append(index)
                occupied_text.add(text_key)
                occupied_hash.add(fingerprint)
                if len(chosen) == n:
                    break
            assert len(chosen) == n
            spec.update(indices=sorted(chosen), doc_sha256=[digest(docs[i]) for i in sorted(chosen)])
        setup['count'] = sum(len(v['indices']) for v in setup['selection'].values())
        verify_tasks(tasks, setup['selection'])
    suite.pop('sha256')
    suite['sha256'] = digest(suite)
    counts = {k:sum(len(v['indices']) for v in g['selection'].values()) for k,g in suite['text'].items()}
    assert counts == {'math500':40,'bbh':54}
    freeze(root/'fresh-suite.json', suite)
    freeze(root/'fresh-selection.json', {'items':counts, 'suite_sha256':suite['sha256'],
           'excluded_sha256':[s['sha256'] for s in history], 'salt':rule['salt']})
    print(suite['sha256'], counts, flush=True)


if __name__ == '__main__':
    protocol_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('protocol.json')
    prepare(Path(sys.argv[1]).resolve(), json.loads(protocol_path.read_text()))
