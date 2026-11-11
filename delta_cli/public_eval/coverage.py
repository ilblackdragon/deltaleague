"""Full coverage of the pinned public suite using the shared CLI scoring adapters."""
from copy import deepcopy


def categories(suite, group):
    multi = group == 'bfcl_multi_turn'
    return [name for name in suite.get('bfcl', {}) if name.startswith('multi_turn_') == multi]


def populations(suite):
    return {**{b: sum(e['population'] for e in s['selection'].values()) for b,s in suite['text'].items()},
            **{b: s['population'] for b,s in suite['code'].items()},
            **{b: sum(suite['bfcl'][c]['population'] for c in categories(suite,b))
               for b in ('bfcl_single_turn','bfcl_multi_turn')}}


def expand(suite, group):
    from delta_cli.public_eval.common import digest
    expanded = deepcopy(suite)
    if group in suite['text']:
        from .text import load_tasks, verify_tasks
        spec = expanded['text'][group]
        tasks = load_tasks(spec['task'], suite['dataset_revisions'])
        verify_tasks(tasks, spec['selection'])
        for name, task in tasks.items():
            entry = spec['selection'][name]
            entry['indices'] = list(range(len(task.eval_docs)))
            entry['doc_sha256'] = [digest(doc) for doc in task.eval_docs]
            entry['count'] = len(task.eval_docs)
        spec['count'] = sum(e['count'] for e in spec['selection'].values())
    elif group in suite['code']:
        import json
        from pathlib import Path
        spec = expanded['code'][group]
        frozen = json.loads((Path(__file__).parent / 'data/full_benchmark_code.json').read_text())[group]
        if frozen['dataset'] != spec['dataset']: raise ValueError('Coding dataset identity')
        docs = frozen['docs']
        if len(docs) != spec['population'] or [docs[i] for i in spec['indices']] != spec['docs']:
            raise ValueError('Coding dataset changed')
        spec.update(docs=docs, indices=list(range(len(docs))), count=len(docs))
    elif group in ('bfcl_single_turn', 'bfcl_multi_turn'):
        from delta_cli.public_eval.tools import verify_data
        from bfcl_eval.utils import load_dataset_entry
        verify_data(suite)
        expanded['bfcl'] = {c: expanded['bfcl'][c] for c in categories(suite,group)}
        for category, spec in expanded['bfcl'].items():
            rows = load_dataset_entry(category)
            by_id = {r['id']:r for r in rows}
            if len(rows) != spec['population'] or [digest(by_id[i]) for i in spec['ids']] != spec['doc_sha256']:
                raise ValueError('BFCL dataset changed')
            spec.update(ids=[r['id'] for r in rows], doc_sha256=[digest(r) for r in rows], count=len(rows))
    else:
        raise ValueError('Unknown benchmark: '+group)
    return expanded


def slice_suite(suite, group, bounds):
    """Stable flattened ordinal partition; preserve native case IDs and hashes."""
    counts = {**{k: sum(len(e['indices']) for e in v['selection'].values()) for k,v in suite['text'].items()},
              **{k: len(v['indices']) for k,v in suite['code'].items()},
              **{g: sum(len(suite.get('bfcl', {})[c]['ids']) for c in categories(suite,g))
                 for g in ('bfcl_single_turn','bfcl_multi_turn')}}
    if (group not in counts or len(bounds) != 2 or any(type(n) is not int for n in bounds)
            or not 0 <= bounds[0] < bounds[1] <= counts[group]):
        raise ValueError('Invalid benchmark shard bounds')
    suite = deepcopy(suite)
    start,end=bounds;position=0
    if group in suite['text']:
        spec=suite['text'][group];selected={}
        for name,entry in spec['selection'].items():
            n=len(entry['indices']);lo=max(0,start-position);hi=min(n,end-position)
            if hi>lo:
                entry.update(indices=entry['indices'][lo:hi],doc_sha256=entry['doc_sha256'][lo:hi],count=hi-lo)
                selected[name]=entry
            position+=n
        spec['selection']=selected;spec['count']=end-start
    elif group in suite['code']:
        spec=suite['code'][group];spec['docs']=spec['docs'][start:end];spec['indices']=spec['indices'][start:end];spec['count']=end-start
    else:
        selected={}
        for name in categories(suite, group):
            entry=suite['bfcl'][name]
            n=len(entry['ids']);lo=max(0,start-position);hi=min(n,end-position)
            if hi>lo:
                entry.update(ids=entry['ids'][lo:hi],doc_sha256=entry['doc_sha256'][lo:hi],count=hi-lo);selected[name]=entry
            position+=n
        suite['bfcl']=selected
    return suite

