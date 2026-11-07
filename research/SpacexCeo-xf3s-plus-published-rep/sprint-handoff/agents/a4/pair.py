"""Paired comparison of two proxy results: acc on prompts closed (and not truncated) in BOTH, flips, early-closes."""
import json, sys
a = {r['id']: r for r in json.load(open(sys.argv[1]))['rows']}
b = {r['id']: r for r in json.load(open(sys.argv[2]))['rows']}
ok = lambda r: r['closed'] and not r['truncated']
both = [i for i in a if ok(a[i]) and ok(b[i])]
print(f'{sys.argv[1]} vs {sys.argv[2]}')
print(' closed in both', len(both), 'acc A', round(100*sum(a[i]['score'] for i in both)/len(both),1), 'acc B', round(100*sum(b[i]['score'] for i in both)/len(both),1))
print(' A-only closed', sum(ok(a[i]) and not ok(b[i]) for i in a), 'acc', round(100*sum(a[i]['score'] for i in a if ok(a[i]) and not ok(b[i]))/max(1,sum(ok(a[i]) and not ok(b[i]) for i in a)),1),
      '| B-only closed', sum(ok(b[i]) and not ok(a[i]) for i in a), 'acc', round(100*sum(b[i]['score'] for i in a if ok(b[i]) and not ok(a[i]))/max(1,sum(ok(b[i]) and not ok(a[i]) for i in a)),1))
print(' wins A', sum(a[i]['score'] > b[i]['score'] for i in a), 'wins B', sum(b[i]['score'] > a[i]['score'] for i in a))
print(' closed-but-truncated (answer overrun) A', sum(a[i]['closed'] and a[i]['truncated'] for i in a), 'B', sum(b[i]['closed'] and b[i]['truncated'] for i in b))
