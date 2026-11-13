"""a6: print (s,g) and (c,g) tables from eval_hazard output json."""
import json, sys
R = json.load(open(sys.argv[1]))
G = [1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
def tab(key, fixed, var, vals, title):
    print(f'\n{title}: {key}   rows={var}, cols=g')
    print('      ' + ' '.join(f'{g:>7}' for g in G))
    for v in vals:
        row = []
        for g in G:
            m = [r for r in R if r[var] == v and r['g'] == g and all(r[k] == x for k, x in fixed.items())]
            row.append(f'{m[0][key]:7.3f}' if m and m[0][key] is not None else '      -')
        print(f'{v:<6}' + ' '.join(row))
S = sorted({r['s'] for r in R if r['c'] == 1.3}); C = sorted({r['c'] for r in R if r['s'] == 0.5})
for k in ['hedge7_per1k', 'hedgeO_per1k', 'entropy']: tab(k, {'c': 1.3}, 's', S, 'close 1.3')
for k in ['close_p_pos0', 'close_at_natural_end', 'close_haz_early_per1k', 'close_haz_trunc_late_per1k']: tab(k, {'s': 0.5}, 'c', C, 's 0.5')
print('\nextra points'); [print(r) for r in R[-10:]]
