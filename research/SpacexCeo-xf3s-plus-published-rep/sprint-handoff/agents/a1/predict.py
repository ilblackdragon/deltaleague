#!/usr/bin/env python3
"""Map proxy result JSON(s) to a predicted PRIVATE test score with an interval and a SUBMIT/CONFIRM/REJECT verdict. ADVISORY.

usage: python predict.py CAND.json [CAND2.json ...] [--ref REF.json ...] [--json]
  Every result file needs its .texts.jsonl next to it (q.py result NAME DIR --texts): strict acc needs thinking lengths.
  Candidate rows from all CAND files (any `samples`, any seed set) are pooled per prompt (id % 10000).
  Reference (default) = pooled b8af1dbc runs with the SAME seed sets as the candidate: proxy400 rows -> res/ref_xf3s_x4.json
  (samples 4) + res/c_xf3s.json; proxy_s2 rows (ids >= 10000) -> res/a1_xf3s_s2.json. Files are independent draws
  (vLLM output depends on batch composition, so equal seeds do not reproduce across runs with different `samples`).

Statistic: strict = 100 * [correct AND closed AND not truncated AND >= 50 thinking tokens] (= a6 strict_score).
  d_i = mean strict over the candidate's samples of prompt i - mean over the reference samples of prompt i
  d = mean_i d_i ; SE = sd(d_i)/sqrt(320) ; z = d/SE.  (Paired by prompt; no common-random-number benefit exists.)
Prediction: anchored = 36.0 + b*d with SE sqrt((b*SE)^2 + (se_b*d)^2 + edit_resid^2); absolute = a + b*strict_cand.
Verdict (calib.json 'rule'): guards (short_think, overflow, acc|healthy) -> REJECT; candidate with < 2 samples per prompt
  -> CONFIRM if d >= screen; else SUBMIT iff d >= confirm_min AND z >= min_z AND anchored >= min_pred_private."""
import json, os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import stats_of, enrich  # noqa: E402

REF_POOL = [os.path.join(HERE, 'res', f) for f in ('ref_xf3s_x4.json', 'c_xf3s.json', 'a1_xf3s_s2.json')]


def rows_of(path):
    if not os.path.exists(path): sys.exit(f'{path} not found')
    d = enrich(os.path.abspath(path))
    if not all('think_tok' in r for r in d['rows']):
        sys.exit(f'{path}: texts file missing ({path[:-5]}.texts.jsonl); fetch with q.py result NAME DIR --texts')
    return d['rows']


def pool(paths):
    """rows from several files. NOT de-duplicated by seed: vLLM outputs depend on batch composition, so sample 0 of a
    samples=4 run does NOT reproduce the samples=1 run with the same seed (21% identical, all of them truncations);
    every file is an independent draw. Only a file passed twice is dropped."""
    out = []
    for p in dict.fromkeys(os.path.abspath(x) for x in paths):
        out.extend(rows_of(p))
    return out


def strict_by_prompt(rows):
    by = {}
    for r in rows:
        ok = r['closed'] and not r['truncated'] and r['think_tok'] >= 50
        by.setdefault(r['id'] % 10000, []).append(100.0 * r['score'] * ok)
    return by


def predict(cands, refs=None):
    C = json.load(open(os.path.join(HERE, 'calib.json')))
    f = C['fits'][C['best']]; R = C['rule']; b = f['b']
    cr = pool(cands)
    if not refs:   # match seed-set composition: proxy400 ids (<10000) -> x4 + c_xf3s ; proxy_s2 ids (>=10000) -> a1_xf3s_s2
        has1, has2 = any(r['id'] < 10000 for r in cr), any(r['id'] >= 10000 for r in cr)
        refs = [p for p in REF_POOL if os.path.exists(p) and ((has2 and 'a1_xf3s_s2' in p) or (has1 and 'a1_xf3s_s2' not in p))]
    rr = pool(refs)
    A, B = strict_by_prompt(cr), strict_by_prompt(rr); ids = sorted(set(A) & set(B))
    d_i = np.array([np.mean(A[i]) - np.mean(B[i]) for i in ids])
    d = float(d_i.mean()); se_x = float(d_i.std(ddof=1) / math.sqrt(len(ids)))
    mu = R['anchor_private'] + b * d
    se = math.sqrt((b * se_x) ** 2 + (f['se_b'] * d) ** 2 + f['edit_resid_sd'] ** 2)
    sc, sr = stats_of(cr), stats_of(rr)
    k_c = len(cr) / len(A); k_r = len(rr) / len(B)
    out = {'stat': C['best'], 'cand_samples_per_prompt': k_c, 'ref_samples_per_prompt': k_r, 'n_prompts': len(ids),
           'd_strict': d, 'd_strict_se': se_x, 'z': d / se_x if se_x else float('nan'),
           'anchored': mu, 'anchored_se': se, 'absolute': f['a'] + b * sc['acc_healthy'], 'absolute_pi80': 1.28 * f['loo_rmse'],
           'cand': {key: sc.get(key) for key in ('acc', 'acc_healthy', 'healthy', 'closed', 'overflow', 'short_think', 'acc_thought', 'tokens')},
           'ref': {key: sr.get(key) for key in ('acc', 'acc_healthy', 'healthy', 'closed', 'overflow', 'short_think', 'acc_thought', 'tokens')},
           'files': cands, 'ref_files': refs}
    cm = C.get('closed_map')
    if cm:
        out['server_closed_est'] = cm['a'] + cm['b'] * sc['closed']; out['via_closed'] = 8.72 + 36.67 * out['server_closed_est'] / 100
    fails = []
    if sc['short_think'] > R['max_short_think']: fails.append(f"short_think {sc['short_think']:.1f}% > {R['max_short_think']}%")
    if sc['overflow'] > R['max_overflow']: fails.append(f"overflow {sc['overflow']:.1f}% > {R['max_overflow']}%")
    dq = sc['acc_thought'] - sr['acc_thought']
    if dq < -R['max_healthy_acc_drop'] and d < 2 * R['screen_min_dstrict']:
        fails.append(f'acc among healthy rows down {dq:.1f} (strict gain may be closing-only)')
    if fails or d < R['screen_min_dstrict']:
        out['decision'] = 'REJECT: ' + ('; '.join(fails) if fails else f'd_strict {d:+.2f} < screen {R["screen_min_dstrict"]}')
    elif k_c < 2:
        out['decision'] = 'CONFIRM: passes screen with 1 sample/prompt; re-run with samples>=3 (or fresh seed sets) before submitting'
    elif d >= R['confirm_min_dstrict'] and out['z'] >= R['min_z'] and mu >= R['min_pred_private']:
        out['decision'] = 'SUBMIT'
    else:
        out['decision'] = (f"REJECT: pooled d_strict {d:+.2f} (z {out['z']:.1f}, pred {mu:.1f}) below confirm thresholds "
                           f"(d >= {R['confirm_min_dstrict']}, z >= {R['min_z']}, pred >= {R['min_pred_private']})")
    return out


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'): print(__doc__); return
    js = '--json' in a; a = [x for x in a if x != '--json']
    if '--ref' in a:
        i = a.index('--ref'); cands, refs = a[:i], a[i + 1:]
    else: cands, refs = a, None
    o = predict(cands, refs)
    if js: print(json.dumps(o, indent=1)); return
    c, r = o['cand'], o['ref']
    print(f"cand ({o['cand_samples_per_prompt']:.1f} samples/prompt): strict {c['acc_healthy']:.1f}  acc {c['acc']:.1f}  healthy {c['healthy']:.1f}%  "
          f"overflow {c['overflow']:.1f}%  short_think {c['short_think']:.1f}%  acc|healthy {c['acc_thought']:.1f}  tokens {c['tokens']:.0f}")
    print(f"ref  ({o['ref_samples_per_prompt']:.1f} samples/prompt): strict {r['acc_healthy']:.1f}  acc {r['acc']:.1f}  healthy {r['healthy']:.1f}%  "
          f"overflow {r['overflow']:.1f}%  short_think {r['short_think']:.1f}%  acc|healthy {r['acc_thought']:.1f}  [{', '.join(os.path.basename(x) for x in o['ref_files'])}]")
    print(f"pooled d_strict {o['d_strict']:+.2f} +- {o['d_strict_se']:.2f}  (z {o['z']:+.1f}, {o['n_prompts']} prompts)")
    print(f"anchored : private ~ {o['anchored']:.1f}  (SE {o['anchored_se']:.1f}; 80% PI {o['anchored']-1.28*o['anchored_se']:.1f}..{o['anchored']+1.28*o['anchored_se']:.1f})")
    print(f"absolute : private ~ {o['absolute']:.1f}  (80% PI +-{o['absolute_pi80']:.1f})")
    if 'via_closed' in o: print(f"secondary via server-closed map: ~{o['via_closed']:.1f} (ignores quality; under-predicts recent leaders; not used)")
    print('decision :', o['decision'])


if __name__ == '__main__':
    main()
