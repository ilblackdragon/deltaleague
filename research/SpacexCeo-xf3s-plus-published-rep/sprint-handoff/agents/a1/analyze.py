"""A1 calibration: proxy statistics vs PRIVATE test score of known leaderboard checkpoints.
usage: python analyze.py            (reads res/*.json, writes calib.json + prints tables)
All proxy numbers are ADVISORY (local vLLM, 320 public prompts); private/trunc are server ground truth (lb.json + brief)."""
import json, os, glob, math, itertools
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'res')
# job -> (version, private, server trunc, server output tokens).  None = not known to us.
KNOWN = {
    'c_xf3s':    ('b8af1dbc', 36.0, None, None),
    'c_new':     ('a9340e14', 31.4, None, None),
    'c_leader':  ('ab592312', 28.2, 0.526, 2954104),
    'c_rep':     ('9ae2205f', 26.2, 0.504, 2840728),
    'c_g175':    ('86968401', 26.0, 0.548, 3025232),
    'c_damp04':  ('67c994de', 22.4, 0.598, 3199501),
    'c_g125':    ('4fe2f3d6', 19.0, 0.756, 3567886),
    'c_base':    ('0ea62f40', 18.6, 0.746, 3583738),
    'c_think15': ('0ffb5aef', 17.6, 0.440, 2063532),
    'c_g1667':   ('d17ce35f', 17.2, 0.826, 3645515),
}
REPLICATES = {'a1_new_s2': 'c_new', 'a1_xf3s_s2': 'c_xf3s'}   # r_newleader was bit-identical to c_new (same weights + seeds => deterministic), so not a noise replicate
MATH = ('math500', 'gsm8k'); MC = ('mmlu_pro', 'arc_c')


def stats_of(rows):
    """All candidate proxy statistics from per-prompt rows (sample-averaged per prompt id first)."""
    n = len(rows)
    sc = np.array([r['score'] for r in rows]); cl = np.array([r['closed'] for r in rows], float)
    tr = np.array([r['truncated'] for r in rows], float); tk = np.array([r['tokens'] for r in rows], float)
    b = np.array([r['bench'] for r in rows]); m = np.isin(b, MATH)
    out = {'acc': 100 * sc.mean(), 'closed': 100 * (1 - tr.mean()), 'trunc': tr.mean(), 'tokens': tk.mean(),
           'acc_closed': 100 * sc[cl > 0].mean() if cl.sum() else float('nan'),
           'math_acc': 100 * sc[m].mean(), 'math_closed': 100 * (1 - tr[m].mean()), 'mc_acc': 100 * sc[~m].mean(),
           'mc_closed': 100 * (1 - tr[~m].mean()), 'n': n}
    for bb in ('math500', 'gsm8k', 'mmlu_pro', 'arc_c'):
        k = b == bb; out[f'{bb}_acc'] = 100 * sc[k].mean(); out[f'{bb}_closed'] = 100 * (1 - tr[k].mean())
    # binomial / sample SE of each statistic (per run, items treated as independent draws)
    out['se_acc'] = 100 * sc.std(ddof=1) / math.sqrt(n)
    out['se_closed'] = 100 * tr.std(ddof=1) / math.sqrt(n)
    out['se_tokens'] = tk.std(ddof=1) / math.sqrt(n)
    out['se_math_acc'] = 100 * sc[m].std(ddof=1) / math.sqrt(m.sum())
    out['se_math_closed'] = 100 * tr[m].std(ddof=1) / math.sqrt(m.sum())
    # answer-phase overflow: </think> emitted but generation still hit max_tokens (answer-phase rambling)
    ov = cl * tr; out['overflow'] = 100 * ov.mean()
    out['closed_think'] = 100 * cl.mean()          # share that emitted </think> at all (incl. overflow)
    if all('think_tok' in r for r in rows):        # needs texts (enrich())
        th = np.array([r['think_tok'] for r in rows], float); at = np.array([r['ans_tok'] for r in rows], float)
        out['short_think'] = 100 * ((th < 50) & (cl > 0)).mean()      # skipped/near-skipped thinking
        out['think_tok_closed'] = float(np.median(th[cl > 0])) if cl.sum() else float('nan')
        out['ans_tok_closed'] = float(np.mean(at[cl > 0])) if cl.sum() else float('nan')
        out['ans_long'] = 100 * ((at > 2048) & (cl > 0)).mean()      # answer phase > 2k tokens
        k = (th >= 50) & (cl > 0) & (tr == 0)
        out['acc_thought'] = 100 * sc[k].mean() if k.sum() else float('nan')   # acc among normally-thought, finished items
        out['acc_healthy'] = 100 * (sc * k).mean()      # acc counting skipped-think and overflow generations as 0
        out['healthy'] = 100 * k.mean()                 # share: thought >=50 tok, closed, finished
    return out


_TOK = None
def enrich(name):
    """Add think_tok / ans_tok per row from the texts file (cached in res/NAME.feat.json)."""
    global _TOK
    d = load(name); rows = d['rows']
    base = name[:-5] if name.endswith('.json') else os.path.join(RES, name)
    cache = base + '.feat.json'; tf = base + '.texts.jsonl'
    if os.path.exists(cache): F = json.load(open(cache))
    elif os.path.exists(tf):
        if _TOK is None:
            from tokenizers import Tokenizer
            _TOK = Tokenizer.from_file('/home/user/delta/exp-base/model/tokenizer.json')
        F = {}
        for line in open(tf):
            t = json.loads(line); txt = t['text']
            if '</think>' in txt:
                th, ans = txt.split('</think>', 1)
                F[f"{t['id']}_{t['sample']}"] = [len(_TOK.encode(th).ids), len(_TOK.encode(ans).ids)]
            else:
                F[f"{t['id']}_{t['sample']}"] = [len(_TOK.encode(txt).ids), 0]
        json.dump(F, open(cache, 'w'))
    else: return d
    for r in rows:
        r['think_tok'], r['ans_tok'] = F[f"{r['id']}_{r['sample']}"]
    return d


def load(name):
    f = name if name.endswith('.json') else os.path.join(RES, f'{name}.json')
    if not os.path.exists(f): return None
    return json.load(open(f))


def loo_rmse(X, y):
    e = []
    for i in range(len(y)):
        k = np.ones(len(y), bool); k[i] = False
        A = np.c_[np.ones(k.sum()), X[k]]; beta = np.linalg.lstsq(A, y[k], rcond=None)[0]
        e.append(y[i] - np.r_[1, X[i]] @ beta)
    return float(np.sqrt(np.mean(np.square(e)))), e


def boot_spearman(x, y, B=4000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) < 3: continue
        r = stats.spearmanr(x[i], y[i])[0]
        if not np.isnan(r): v.append(r)
    return np.percentile(v, [5, 95])


def main():
    names = [k for k in KNOWN if load(k)]
    S = {k: stats_of(enrich(k)['rows']) for k in names}
    if 'c_xf3s' in S and load('ref_xf3s_x4'):   # pool the anchor: c_xf3s (1 sample) + ref_xf3s_x4 (4 samples), independent draws
        S['c_xf3s'] = stats_of(enrich('c_xf3s')['rows'] + enrich('ref_xf3s_x4')['rows'])
    y = np.array([KNOWN[k][1] for k in names])
    print(f'{len(names)} calibration runs available: {names}\n')
    cols = ['acc', 'closed', 'acc_closed', 'tokens', 'math_acc', 'mc_acc', 'math500_acc', 'gsm8k_acc', 'mmlu_pro_acc', 'arc_c_acc', 'overflow', 'short_think', 'think_tok_closed', 'ans_tok_closed', 'acc_thought', 'acc_healthy', 'healthy']
    print('job        private srv_trunc | ' + ' '.join(f'{c[:11]:>11s}' for c in cols))
    for k in sorted(names, key=lambda k: -KNOWN[k][1]):
        st = KNOWN[k][2]; print(f'{k:10s} {KNOWN[k][1]:6.1f} {("%.3f" % st) if st is not None else "  ?  ":>9s} | ' + ' '.join(f'{S[k].get(c, float("nan")):11.1f}' for c in cols))
    print('\nper-run SE (pts): ' + ', '.join(f"{k}: acc {S[k]['se_acc']:.2f} closed {S[k]['se_closed']:.2f} tok {S[k]['se_tokens']:.0f}" for k in names[:3]))

    # proxy closed vs server closed
    srv = [k for k in names if KNOWN[k][2] is not None]; closed_map = None
    if len(srv) >= 3:
        xs = np.array([S[k]['closed'] for k in srv]); ys = np.array([100 * (1 - KNOWN[k][2]) for k in srv])
        b = np.polyfit(xs, ys, 1); closed_map = {'a': float(b[1]), 'b': float(b[0]), 'n': len(srv), 'r': float(np.corrcoef(xs, ys)[0, 1])}
        print(f'\nserver closed% ~ {b[1]:.1f} + {b[0]:.2f} * proxy closed%   r={np.corrcoef(xs, ys)[0,1]:+.3f}  (n={len(srv)})')

    fits = {}
    if len(names) >= 4:
        print('\nsingle-statistic fits  private ~ a + b*x  (LOO = leave-one-checkpoint-out prediction RMSE; Spearman 90% bootstrap CI)')
        cands = ['acc', 'closed', 'acc_closed', 'tokens', 'math_acc', 'math_closed', 'mc_acc', 'math500_acc', 'gsm8k_acc', 'mmlu_pro_acc', 'arc_c_acc', 'math500_closed', 'gsm8k_closed', 'overflow', 'short_think', 'think_tok_closed', 'acc_thought', 'acc_healthy', 'healthy']
        for c in cands:
            x = np.array([S[k].get(c, np.nan) for k in names])
            if np.isnan(x).any(): continue
            b = np.polyfit(x, y, 1)
            l, e = loo_rmse(x[:, None], y); rho = stats.spearmanr(x, y)[0]; ci = boot_spearman(x, y)
            kt = stats.kendalltau(x, y)[0]
            fits[c] = {'a': float(b[1]), 'b': float(b[0]), 'loo_rmse': l, 'r': float(np.corrcoef(x, y)[0, 1]), 'spearman': float(rho),
                       'kendall': float(kt), 'spearman_ci90': [float(ci[0]), float(ci[1])], 'resid_sd': float(np.std(y - np.polyval(b, x), ddof=2)),
                       'loo_errors': {k: float(v) for k, v in zip(names, e)},
                       'se_b': float(np.sqrt(np.sum((y - np.polyval(b, x))**2) / (len(y) - 2) / np.sum((x - x.mean())**2)))}
            # residual not explained by server noise (SE 2.0) and proxy noise (~2.6 pts per run, times b)
            fits[c]['edit_resid_sd'] = float(np.sqrt(max(1.5**2, fits[c]['resid_sd']**2 - 2.0**2 - (b[0] * 2.6)**2)))
            print(f'  {c:15s} a {b[1]:7.2f} b {b[0]:8.4f}  r {fits[c]["r"]:+.3f}  rho {rho:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] tau {kt:+.2f}  LOO {l:5.2f}')
        # ratio model, no intercept
        x = np.array([S[k]['acc'] for k in names]); kk = float((x @ y) / (x @ x)); e = []
        for i in range(len(y)):
            m = np.arange(len(y)) != i; e.append(y[i] - x[i] * (x[m] @ y[m]) / (x[m] @ x[m]))
        fits['ratio_acc'] = {'k': kk, 'loo_rmse': float(np.sqrt(np.mean(np.square(e))))}
        print(f'  private = k*acc (= k*(1-trunc)*acc_closed):  k {kk:.3f}  LOO {fits["ratio_acc"]["loo_rmse"]:.2f}')
        # two-factor model closed + acc_closed (3 params, only reported with LOO)
        for pair in [('closed', 'acc_closed'), ('math_closed', 'math_acc'), ('closed', 'acc')]:
            X = np.array([[S[k][pair[0]], S[k][pair[1]]] for k in names]); l, _ = loo_rmse(X, y)
            beta = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0]
            fits['+'.join(pair)] = {'beta': beta.tolist(), 'loo_rmse': l}
            print(f'  private ~ {beta[0]:.2f} + {beta[1]:.3f}*{pair[0]} + {beta[2]:.3f}*{pair[1]}   LOO {l:5.2f}')
        print(f'  null model (mean)  LOO {np.sqrt(np.mean([(y[i] - np.delete(y, i).mean())**2 for i in range(len(y))])):.2f}')

    # run-to-run noise from replicates
    rep = {}
    for r, base in REPLICATES.items():
        a, bse = load(r), load(base)
        if not (a and bse): continue
        sa, sb = stats_of(a['rows']), stats_of(bse['rows'])
        ra = {x['id'] % 10000: x for x in a['rows']}; rb = {x['id'] % 10000: x for x in bse['rows']}
        ids = sorted(set(ra) & set(rb))
        pa = np.array([ra[i]['score'] for i in ids]); pb = np.array([rb[i]['score'] for i in ids])
        ta = np.array([ra[i]['truncated'] for i in ids], float); tb = np.array([rb[i]['truncated'] for i in ids], float)
        rep[r] = {'vs': base, 'd_acc': sa['acc'] - sb['acc'], 'd_closed': sa['closed'] - sb['closed'], 'd_tokens': sa['tokens'] - sb['tokens'],
                  'corr_score': float(np.corrcoef(pa, pb)[0, 1]), 'corr_trunc': float(np.corrcoef(ta, tb)[0, 1]),
                  'paired_se_acc_diff': float(100 * (pa - pb).std(ddof=1) / math.sqrt(len(ids))),
                  'paired_se_closed_diff': float(100 * (ta - tb).std(ddof=1) / math.sqrt(len(ids))),
                  'same_score_frac': float((pa == pb).mean())}
        print(f'\nreplicate {r} vs {base}: ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep[r].items()}))
    best = min((c for c in fits if 'b' in fits[c]), key=lambda c: fits[c]['loo_rmse']) if fits else None
    print(f'\nbest single statistic by LOO: {best}' + (f"  se_b {fits[best]['se_b']:.3f}  edit_resid_sd {fits[best]['edit_resid_sd']:.2f}" if best else ''))
    rule = {'anchor': 'c_xf3s', 'anchor_private': 36.0, 'screen_min_dstrict': 3.5, 'confirm_min_dstrict': 4.5, 'max_short_think': 3.0,
            'max_overflow': 17.0, 'max_healthy_acc_drop': 5.0, 'min_pred_private': 38.3, 'min_z': 2.5}
    json.dump({'best': best, 'rule': rule, 'closed_map': closed_map, 'names': names, 'private': y.tolist(), 'stats': S, 'fits': fits, 'replicates': rep}, open(os.path.join(HERE, 'calib.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
