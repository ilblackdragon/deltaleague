import json, sys, statistics as st, collections
A = json.load(open(sys.argv[1]))
def pct(x, n): return f'{x}/{n} ({100*x/max(1,n):.1f}%)'
nc = [a for a in A if a['closed'] and not a['truncated']]   # closed and answered within budget
tr = [a for a in A if a['truncated']]
trn = [a for a in tr if not a['closed']]                    # never closed
trc = [a for a in tr if a['closed']]                        # closed, but answer cut by 8192
print('n', len(A), 'closed_ok', len(nc), 'trunc', len(tr), 'trunc_noclose', len(trn), 'trunc_closed_late', len(trc),
      'acc trunc_closed_late', sum(a['score'] for a in trc))
for name, G in [('trunc_noclose', trn), ('closed_ok', nc)]:
    print(f'== {name} n={len(G)}')
    for b in ['ALL', 'math500', 'gsm8k', 'mmlu_pro', 'arc_c']:
        g = G if b == 'ALL' else [a for a in G if a['bench'] == b]
        if not g: continue
        hs = [a for a in g if a['n_stmts']]
        anyc = [a for a in g if a.get('any_correct')]
        fc = [a for a in hs if a['first_stmt_ok']]
        print(f' {b:8s} n={len(g):3d} has_stmt {pct(len(hs),len(g))} any_correct {pct(len(anyc),len(g))} first_stmt_correct {pct(len(fc),len(g))}'
              + (f" med first_stmt_tok {st.median(a['first_stmt_tok'] for a in hs):.0f}" if hs else '')
              + (f" med first_correct_tok {st.median(a['first_correct_tok'] for a in anyc):.0f}" if anyc else ''))
    if name == 'closed_ok':
        hs = [a for a in G if a['n_stmts']]
        print(' closed: acc', sum(a['score'] for a in G) / len(G), ' with stmt: first_ok&score', sum(a['first_stmt_ok'] and a['score'] > 0 for a in hs),
              'first_wrong&score', sum((not a['first_stmt_ok']) and a['score'] > 0 for a in hs), 'first_ok&fail', sum(a['first_stmt_ok'] and a['score'] == 0 for a in hs),
              'no stmt & score', sum(a['score'] > 0 for a in G if not a['n_stmts']), 'no stmt', len(G) - len(hs))
        print(' closed: median think tok', st.median(a['think_tokens'] for a in G),
              ' tokens after last stmt (median)', st.median(a['think_tokens'] - a['last_stmt_tok'] for a in hs),
              ' after first stmt', st.median(a['think_tokens'] - a['first_stmt_tok'] for a in hs))
g = trn
print('== tail structure of never-closed traces')
print(' exact period found (<=1200 tok, 300-tok check):', pct(sum(a['period'] is not None for a in g), len(g)),
      ' periods:', sorted(a['period'] for a in g if a['period']))
print(' dup_line_frac median', st.median(a['dup_line_frac'] for a in g), ' >0.5:', pct(sum(a['dup_line_frac'] > 0.5 for a in g), len(g)))
print(' zlib ratio median', round(st.median(a['zlib_ratio'] for a in g), 3), ' tail hedges/1500tok median', st.median(a['tail_hedges'] for a in g))
anyc = [a for a in g if a.get('any_correct')]
print(' correct-stated traces: frac tokens after first correct', st.median((8192 - a['first_correct_tok']) / 8192 for a in anyc) if anyc else None)
print(' among correct-stated never-closed: periodic', sum(a['period'] is not None for a in anyc), 'dup>0.5', sum(a['dup_line_frac'] > 0.5 for a in anyc))
hist = collections.Counter(min(7, a['first_correct_tok'] // 1024) for a in anyc)
print(' first_correct_tok histogram (k tokens):', sorted(hist.items()))
