"""a6: filler-phrase rates per 1k generated tokens in the thinking part, split by closed / never-closed traces."""
import json, re, collections, sys
pats = {'Wait': r'\b[Ww]ait\b', 'Hmm': r'\b[Hh]mm\b', 'Maybe': r'\b[Mm]aybe\b', 'Actually': r'\b[Aa]ctually\b', 'Alternatively': r'\b[Aa]lternatively\b',
        "Let's re-": r"Let's re-?(check|evaluate|verify|examine|read|calculate|think|consider|do)", 'But': r'\bBut\b', 'No.': r'(?:^|\n)\s*No[.,]',
        'double-check': r'[Dd]ouble[- ]check', 'Is it/this': r'\bIs (it|this|that)\b', 'Perhaps': r'\b[Pp]erhaps\b', 'However': r'\bHowever\b'}
for name in sys.argv[1:]:
    R = {(r['id'], r['sample']): r for r in json.load(open(f'results/{name}.json'))['rows']}
    agg = collections.defaultdict(collections.Counter); ntok = collections.Counter()
    for l in open(f'results/{name}.texts.jsonl'):
        t = json.loads(l); r = R[(t['id'], t['sample'])]; k = 'closed' if r['closed'] else 'never'
        th = t['text'].split('</think>')[0]; ntok[k] += r['tokens']
        for n, p in pats.items(): agg[k][n] += len(re.findall(p, th))
    for k in ['closed', 'never']: print(name, k, {n: round(1000 * v / max(1, ntok[k]), 2) for n, v in agg[k].items()})
