"""a6: collect every finished queue job (q.py status DONE list) into grid.csv with its edit parameters. Proxy numbers are advisory."""
import json, os, subprocess, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable; T = '/home/user/delta/tools'
os.makedirs(f'{HERE}/results', exist_ok=True); os.makedirs(f'{HERE}/specs_seen', exist_ok=True)
VER = {  # verified by diffing weights vs baseline (rowdiff.py); private = leaderboard test score
 '0ea62f40': dict(s=1, g=1, close=1, mode='-', tokens='none', private=18.6),
 'ab592312': dict(s=0.6, g=1.4, close=1, mode='mean', tokens='11 hedge', private=28.2),
 '86968401': dict(s=0.6, g=1.25, close=1, mode='mean', tokens='11 hedge', private=26.0),
 '9ae2205f': dict(s=0.6, g=1.4, close=1, mode='mean', tokens='11 hedge + rep-dir', private=26.2),
 '8dc7d17c': dict(s=0.4, g=1, close=1, mode='scale', tokens='7 wait/hmm', private=24.6),
 '67c994de': dict(s=0.4, g=1, close=1, mode='mean', tokens='3 (Wait,ĠWait,Hmm)', private=22.4),
 '4fe2f3d6': dict(s=1, g=1.25, close=1, mode='-', tokens='none', private=19.0),
 'd17ce35f': dict(s=1, g=1.667, close=1, mode='-', tokens='none', private=17.2),
 '0ffb5aef': dict(s=1, g=1, close=1.5, mode='-', tokens='</think> only', private=17.6),
 'a155dc2f': dict(s=1, g=1, close=1, mode='-', tokens='sparse MLP', private=19.0),
 'a9340e14': dict(s=0.5, g=1.4, close=1.3, mode='scale', tokens='7 wait/hmm', private=31.4),
 'b8af1dbc': dict(s=0.5, g=1.4, close=1.3, mode='scale', tokens='7 wait/hmm + doubt-dir L0-13 + L16 n998 x-2 (xf3s)', private=36.0)}
st = subprocess.run([PY, f'{T}/q.py', 'status'], capture_output=True, text=True).stdout
done = st.split('DONE:')[1].split('FAILED:')[0].split()
done = [d[:-5] for d in done if d.endswith('.json')]
out = []
for n in sorted(done):
    rf = f'{HERE}/results/{n}.json'
    if not os.path.exists(rf): subprocess.run([PY, f'{T}/q.py', 'result', n, f'{HERE}/results'], capture_output=True)
    sf = f'{HERE}/specs_seen/{n}.json'
    if not os.path.exists(sf): subprocess.run([PY, f'{T}/gpu.py', 'get', f'dl/done/{n}.json', sf], capture_output=True)
    if not os.path.exists(rf): continue
    R = json.load(open(rf)); S = R['summary']; sp = json.load(open(sf)) if os.path.exists(sf) else {}
    if 'model' in sp:
        v = sp['model'].split('/')[1][:8] if sp['model'].startswith('versions/') else sp['model']
        d = dict(VER.get(v, dict(s='', g='', close='', mode='?', tokens=sp['model'], private='')))
        if 'prompts' in sp: d['tokens'] += f" [prompts {sp['prompts']}]"
    elif 'base' in sp and sp['base'].startswith('versions/') and sp['base'].split('/')[1][:8] in VER:
        v = sp['base'].split('/')[1][:8]; d0 = VER[v]; sc = sp.get('scale_tokens', {}); tk = sp.get('tokens', {})
        rs = sorted(set(sc.values())) or [1]
        d = dict(s=round(d0['s'] * rs[0], 4) if len(rs) == 1 else '/'.join(str(round(d0['s'] * x, 4)) for x in rs),
                 g=round(d0['g'] * sp.get('gain', 1), 4), close=round(d0['close'] * sp.get('close_gain', 1), 4),
                 mode=f"rel on {v}: s x{rs[0] if len(rs) == 1 else rs} c x{sp.get('close_gain', 1)} g x{sp.get('gain', 1)}"
                      + (f" +rep{sp['rep_alpha']}" if sp.get('rep_alpha') else '') + (' +tokens' if tk else ''),
                 tokens=d0['tokens'] + (f" + {len(sc)} scaled rows" if sc and set(sc) - {' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm'} else '')
                        + (f" [prompts {sp['prompts']}]" if 'prompts' in sp else ''), private='')
    else:
        tk = sp.get('tokens', {}); sc = sp.get('scale_tokens', {})
        vals = set(tk.values()) | set(sc.values())
        d = dict(s='/'.join(map(str, sorted(vals))) if vals else 1, g=sp.get('gain', 1), close=sp.get('close_gain', 1),
                 mode=('scale' if sc and not tk else sp.get('mode', 'mean') if tk else '-') + (f"+bias{sp['bias']}" if sp.get('bias') else '') + (f"+rep{sp['rep_alpha']}" if sp.get('rep_alpha') else ''),
                 tokens=f'{len(tk) + len(sc)} rows: ' + ','.join(repr(k) for k in list(tk) + list(sc)), private='')
    def pool(bs, key='acc'):
        n_ = sum(S[b]['n'] for b in bs if b in S); return round(sum(S[b][key] * S[b]['n'] for b in bs if b in S) / n_, 2) if n_ else ''
    # skip-thinking share (<50 thinking tokens) and answer-phase overflow (closed but hit max_tokens), needs texts
    tf = f'{HERE}/results/{n}.texts.jsonl'
    if not os.path.exists(tf): subprocess.run([PY, f'{T}/q.py', 'result', n, f'{HERE}/results', '--texts'], capture_output=True)
    short = ''
    if os.path.exists(tf):
        from transformers import AutoTokenizer
        global TOK
        try: TOK
        except NameError: TOK = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
        th = [json.loads(l)['text'] for l in open(tf)]
        short = round(sum(1 for t in th if '</think>' in t and len(TOK(t.split('</think>')[0], add_special_tokens=False)['input_ids']) < 50) / len(th), 3)
    rws = R['rows']; overflow = round(sum(1 for r in rws if r['closed'] and r['truncated']) / len(rws), 3)
    strict = ''
    if os.path.exists(tf):
        sys.path.insert(0, HERE); import strict_score
        cwd = os.getcwd(); os.chdir(HERE); strict = strict_score.regimes(n)[1]; os.chdir(cwd)
    out.append(dict(strict_acc=strict, think_lt50=short, closed_overflow=overflow, name=n, s=d['s'], g=d['g'], close=d['close'], mode=d['mode'], tokens=d['tokens'],
                    acc=round(S['ALL']['acc'], 2), trunc=round(S['ALL']['trunc'], 3), acc_closed=round(S['ALL']['acc_closed'], 2),
                    tokens_mean=round(S['ALL']['tokens']), math_acc=pool(['math500', 'gsm8k']), mc_acc=pool(['mmlu_pro', 'arc_c']),
                    hedges=round(S['ALL']['hedges'], 2), n=S['ALL']['n'], private=d['private']))
cols = ['name', 's', 'g', 'close', 'mode', 'tokens', 'acc', 'strict_acc', 'trunc', 'acc_closed', 'tokens_mean', 'math_acc', 'mc_acc', 'hedges', 'think_lt50', 'closed_overflow', 'n', 'private']
with open(f'{HERE}/grid.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow(r) for r in out]
for r in out: print(','.join(str(r[c]) for c in cols))
