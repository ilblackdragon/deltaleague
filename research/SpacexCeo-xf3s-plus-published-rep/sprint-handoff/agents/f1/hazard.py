"""F1 hazard model on teacher-forced positions.
For an edit theta: pi_t = exact sampler P(</think> at t) ; eta_t = P(hedge token at t).
Per trace: survival S_t = prod_{u<t}(1-pi_u); P(close at t) = pi_t S_t.
Labels: 'premature' if t < A (A = first position where the item's reference answer / trace's own final answer appears
in the thinking text), else 'legit'. Trace-level prediction of closing by 8192 and expected score."""
import sys, os, json, re, numpy as np; sys.path.insert(0, '/home/user/delta/agents/f1')
from sampler import *
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
P = {p['id']: p for p in json.load(open('/home/user/delta/agents/f1/proxy400.json'))}
def answer_pos(t, p, text):
    """position (token index) where the reference answer first appears in the thinking; None if never"""
    ans = str(p['answer']).strip()
    think = text.split('</think>')[0]
    pats = [ans]
    if p['scorer'] != 'math': pats += [f'({ans})', f'{ans}.', f'{ans})', f'option {ans}', f'answer is {ans}']
    best = None
    for pat in pats:
        if p['scorer'] != 'math' and len(ans) == 1:
            m = re.search(r'(?<![A-Za-z])' + re.escape(ans) + r'(?![A-Za-z])', think)
            k = m.start() if m else -1
        else:
            k = think.find(pat)
        if k >= 0 and (best is None or k < best): best = k
    if best is None: return None
    return len(tok(think[:best], add_special_tokens=False)['input_ids'])
def gateway_mask(t):
    """positions right after a '\n' (198) token: where natural closes happen"""
    g = np.zeros(t.n, bool); g[1:] = t.gid[:-1] == 198; return g
def analyze(tr, texts, th, a0=0.31, a_m=0.64, lam_stuck=None, verbose=False):
    out = []
    for k, t in tr.items():
        pid = int(k.split('_')[0]); p = P[pid]; text = texts[pid]
        o = t.probs(th); pi = o['p_close'].copy()
        n = t.n
        A = answer_pos(t, p, text)
        prem = np.ones(n, bool)
        if A is not None: prem[A:] = False
        # survival and close distribution over observed positions
        S = np.concatenate([[1.0], np.cumprod(1 - pi)])[:n]
        pc = pi * S                                  # P(close exactly at t)
        p_prem = pc[prem].sum(); p_legit = pc[~prem].sum(); S_end = S[-1] * (1 - pi[-1])
        out.append(dict(id=pid, bench=p['bench'], n=n, closed=t.closed, correct=texts_score[pid], A=A, p_prem=p_prem, p_legit=p_legit, S_end=S_end,
                        sum_pi=pi.sum(), sum_h7=o['p_h7'].sum(), sum_h13=o['p_h13'].sum(), sum_subs=o['p_subs'].sum(), H=o['H'].mean(),
                        lp_act=np.log(np.maximum(o['p_act'], 1e-12)).mean(), pi_last=pi[-1]))
    return out
def summarize(res, a0=0.31, a_m=0.64, lam_stuck=1e-5):
    """trace-level prediction: closed traces: legit closes keep their observed correctness; premature -> a0;
    mass S_end continues: closes within remaining budget w.p. 1-exp(-lam_stuck*(8192-n)) with accuracy a_m."""
    sc, tr_, n = 0, 0, 0
    for r in res:
        rem = max(0, 8192 - r['n']); p_cont_close = (1 - np.exp(-lam_stuck * rem)) if not r['closed'] or rem > 0 else 0
        acc_legit = r['correct'] if r['closed'] else a_m
        s = r['p_prem'] * a0 + r['p_legit'] * acc_legit + r['S_end'] * p_cont_close * a_m
        t_ = r['S_end'] * (1 - p_cont_close)
        sc += s; tr_ += t_; n += 1
    return 100 * sc / n, tr_ / n
if __name__ == '__main__':
    d = sys.argv[1]; res_name = sys.argv[2]
    tr = load_dir(d)
    texts = {json.loads(l)['id']: json.loads(l)['text'] for l in open(f'/home/user/delta/agents/f1/results/{res_name}.texts.jsonl')}
    rows = {r['id']: r for r in json.load(open(f'/home/user/delta/agents/f1/results/{res_name}.json'))['rows']}
    texts_score = {i: r['score'] for i, r in rows.items()}
    THETAS = {'base': BASE, 'newleader': NEWLEADER, 'leader1': LEADER1, 'g1.4': dict(g=1.4), 'g1.667': dict(g=1.667), 'c1.5': dict(c=1.5),
              'damp7': dict(s={k: 0.5 for k in H7}), 'damp7+g1.4': dict(g=1.4, s={k: 0.5 for k in H7}),
              'stack c1.15': dict(NEWLEADER, c=1.15), 'stack c1.2': dict(NEWLEADER, c=1.2), 'stack c1.4': dict(NEWLEADER, c=1.4), 'stack c1.5': dict(NEWLEADER, c=1.5),
              'stack g1.3': dict(NEWLEADER, g=1.3), 'stack g1.55': dict(NEWLEADER, g=1.55), 'stack h13': dict(NEWLEADER, s={k: 0.5 for k in H13}),
              'stack c1.3 b-0.1': dict(NEWLEADER, bias={'</think>': -0.1}), 'stack c1.3 b+0.1': dict(NEWLEADER, bias={'</think>': 0.1}),
              'stack c1.6 b+0.2': dict(NEWLEADER, c=1.6, bias={'</think>': 0.2}), 'stack c1.6 b+0.3': dict(NEWLEADER, c=1.6, bias={'</think>': 0.3}),}
    print(f'{len(tr)} traces; columns: sum pi (expected closes over observed positions), P(prem close), P(legit close), S_end, E[h7], E[h13], E[subs], mean entropy, mean log p(actual)')
    for name, th in THETAS.items():
        res = analyze(tr, texts, th)
        agg = {k: np.mean([r[k] for r in res]) for k in ['sum_pi', 'p_prem', 'p_legit', 'S_end', 'sum_h7', 'sum_h13', 'sum_subs', 'H', 'lp_act']}
        cl = [r for r in res if r['closed']]; tu = [r for r in res if not r['closed']]
        print(f"{name:18s} sum_pi {agg['sum_pi']:.3f} prem {agg['p_prem']:.3f} legit {agg['p_legit']:.3f} S_end {agg['S_end']:.3f} | closed-traces: prem {np.mean([r['p_prem'] for r in cl]):.3f} legit {np.mean([r['p_legit'] for r in cl]):.3f} pi_last {np.mean([r['pi_last'] for r in cl]):.3f} | trunc-traces: sum_pi {np.mean([r['sum_pi'] for r in tu]):.3f} prem {np.mean([r['p_prem'] for r in tu]):.3f} | h7 {agg['sum_h7']:.1f} h13 {agg['sum_h13']:.1f} subs {agg['sum_subs']:.1f} H {agg['H']:.3f} lp {agg['lp_act']:.3f}")
