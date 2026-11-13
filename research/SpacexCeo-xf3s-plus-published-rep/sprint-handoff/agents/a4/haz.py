"""Hazard (stopping-time) model of </think> on teacher-forced traces of a parent model.
Coupling argument: the stream of non-close tokens has the same law under the parent and under an edit that only changes
the </think> logit (up to nucleus effects), so an edit raising p_t -> p'_t adds an extra hazard
  h_t = (p'_t - p_t) / (1 - p_t)          at every position t before the parent's own stopping time T.
Edited outcome: early close at t with prob h_t * prod_{s<t}(1-h_s), value V(t) = probe correctness at the last probe <= t
(0 if the answer section could not fit: t > 8192 - LANS); else the parent's own outcome (score if closed, 0 if truncated).
Sampling exactly as vLLM: presence penalty (stored in `top`), temperature 1, top_k 20, top_p 0.95.
Library + CLI: haz.py NPZ_DIR RESULT.json  -> prints baseline sanity and a grid of row-scale / constant-bias edits."""
import numpy as np, glob, os, json, sys
LANS = 600
def p_close(lc, top, k=20, p=0.95):
    """lc: (T,), top: (T,24) other tokens' penalised logits, desc. returns sampling prob of </think> per position"""
    L = np.concatenate([lc[:, None], top], 1)              # col 0 = close
    order = np.argsort(-L, 1)[:, :k]
    Lk = np.take_along_axis(L, order, 1)
    pr = np.exp(Lk - Lk[:, :1]); pr /= pr.sum(1, keepdims=True)
    asc = pr[:, ::-1]; cum = np.cumsum(asc, 1)
    keep_asc = cum > 1 - p; keep_asc[:, -1] = True
    keep = keep_asc[:, ::-1]
    pk = pr * keep; pk /= pk.sum(1, keepdims=True)
    is_c = order == 0
    return (pk * is_c).sum(1)

def load(npz_dir, result_json):
    rows = {f"{r['id']}_{r['sample']}": r for r in json.load(open(result_json))['rows']}
    D = []
    for f in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
        k = os.path.basename(f)[:-4]; z = np.load(f)
        r = rows[k]
        d = {'key': k, 'bench': r['bench'], 'score': r['score'], 'closed': bool(z['closed']), 'truncated': r['truncated'],
             'lc': z['lc'], 'top': z['top'], 'hn': z['hn'], 'gid': z['gid'], 'ppos': z['probe_pos'], 'pok': z['probe_ok']}
        D.append(d)
    return D

def value_curve(d):
    T = len(d['lc']); V = np.zeros(T)
    for pos, ok in zip(d['ppos'], d['pok']):
        V[pos:] = float(ok)
    V[max(0, 8192 - LANS):] = 0.0
    return V

def evaluate(D, lc_fn, verbose=False):
    """lc_fn(d) -> new close logits (T,). Returns dict of expected acc, trunc, early-loss."""
    tot = dict(acc=0.0, trunc=0.0, acc0=0.0, trunc0=0.0, gain=0.0, loss=0.0, n=len(D), early=0.0, mean_tok_saved=0.0)
    for d in D:
        lc0, top = d['lc'], d['top']
        T = len(lc0) - 1 if d['closed'] else len(lc0)       # positions strictly before the natural close / end
        p0 = p_close(lc0[:T], top[:T]); p1 = p_close(lc_fn(d)[:T], top[:T])
        h = np.clip((p1 - p0) / np.maximum(1e-12, 1 - p0), 0, 1)
        S = np.concatenate([[1.0], np.cumprod(1 - h)])
        q = h * S[:-1]
        V = value_curve(d)[:T]
        nat = d['score'] if (d['closed'] and not d['truncated']) else 0.0
        e = (q * V).sum() + S[-1] * nat
        tot['acc'] += e; tot['acc0'] += nat
        tot['trunc0'] += float(d['truncated']); tot['trunc'] += S[-1] * float(d['truncated']) + (q * (np.arange(T) > 8192 - LANS)).sum()
        tot['early'] += 1 - S[-1]
        tot['gain'] += (q * V).sum() * (1 - nat) ; tot['loss'] += (1 - S[-1]) * nat - (q * V).sum() * nat
        tot['mean_tok_saved'] += (q * (T - np.arange(T))).sum()
    n = len(D)
    return {k: (v / n if k != 'n' else v) for k, v in tot.items()}

if __name__ == '__main__':
    D = load(sys.argv[1], sys.argv[2])
    print('traces', len(D), 'closed', sum(d['closed'] for d in D))
    for c in [1.0, 1.05, 1.1, 1.154, 1.2, 1.3]:
        print('row x', c, {k: round(v, 4) for k, v in evaluate(D, lambda d, c=c: d['lc'] * c).items()})
    for b in [0.25, 0.5, 1.0, 1.5]:
        print('const +', b, {k: round(v, 4) for k, v in evaluate(D, lambda d, b=b: d['lc'] + b).items()})

def evaluate2(D, lc_fn, rho=0.85, nu=0.15):
    """Like evaluate(), but also handles edits that LOWER p at the natural close position T of closed traces:
    the edited model closes at T with prob min(1, p'_T/p_T); otherwise it continues, valued rho*score for traces that
    were answered within budget and nu for answer-overrun traces (closed but truncated, score 0). Positions t<T where
    p' < p are exact under the coupling (the parent did not close there)."""
    acc = trunc = early = sup = 0.0
    per = []
    for d in D:
        lc0, top = d['lc'], d['top']; lc1 = lc_fn(d)
        T = len(lc0) - 1 if d['closed'] else len(lc0)
        p0 = p_close(lc0[:T], top[:T]); p1 = p_close(lc1[:T], top[:T])
        h = np.clip((p1 - p0) / np.maximum(1e-12, 1 - p0), 0, 1)
        S = np.concatenate([[1.0], np.cumprod(1 - h)]); q = h * S[:-1]
        V = value_curve(d)[:T]
        e = (q * V).sum(); tr_ = (q * (np.arange(T) > 8192 - LANS)).sum()
        if d['closed']:
            pc0 = p_close(lc0[T:T + 1], top[T:T + 1])[0]; pc1 = p_close(lc1[T:T + 1], top[T:T + 1])[0]
            keep = min(1.0, pc1 / max(pc0, 1e-12))
            ok_budget = not d['truncated']
            nat = d['score'] if ok_budget else 0.0
            cont = rho * d['score'] if ok_budget else nu
            e += S[-1] * (keep * nat + (1 - keep) * cont)
            tr_ += S[-1] * keep * float(d['truncated'])
            sup += S[-1] * (1 - keep)
        else:
            tr_ += S[-1]
        early += 1 - S[-1]
        acc += e; trunc += tr_; per.append(e)
    n = len(D)
    return {'acc': acc / n, 'trunc': trunc / n, 'early': early / n, 'suppressed': sup / n,
            'acc0': np.mean([d['score'] if (d['closed'] and not d['truncated']) else 0 for d in D]),
            'trunc0': np.mean([float(d['truncated']) for d in D])}
