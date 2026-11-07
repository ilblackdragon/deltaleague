"""Characterise thinking traces: where an answer is first stated, whether it is correct, tail structure of truncated traces.
usage: ana.py RESULT.json TEXTS.jsonl PROMPTS.json OUT.json"""
import json, re, sys, zlib, collections
from transformers import AutoTokenizer
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
tok = AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
res, texts, prompts, outf = sys.argv[1:5]
rows = {(r['id'], r['sample']): r for r in json.load(open(res))['rows']}
P = {p['id']: p for p in json.load(open(prompts))}
HEDGE = re.compile(r'\b(Wait|wait|Hmm|hmm|Maybe|maybe|Actually|actually|Alternatively|alternatively)\b')

def boxed(s):
    out = []
    for m in re.finditer(r'\\boxed\{', s):
        i = m.end(); d = 1; j = i
        while j < len(s) and d:
            d += {'{': 1, '}': -1}.get(s[j], 0); j += 1
        if d == 0: out.append((m.start(), j, s[i:j - 1]))
    return out

MC_PAT = re.compile(r'(?:answer is|Answer:|answer:|correct option is|correct answer is|option)\s*\**\s*\(?([A-J])\b\)?')
MATH_PAT = re.compile(r'(?:the answer is|final answer is|Answer:)\s*\$?([^\n$]{1,40})')

def stmts(think, p):
    """answer statements (char_pos, value, correct)"""
    out = []
    for a, b, v in boxed(think):
        out.append((a, v))
    if p['scorer'] == 'mc':
        for m in MC_PAT.finditer(think): out.append((m.start(), m.group(1)))
    else:
        for m in MATH_PAT.finditer(think): out.append((m.start(), m.group(1).strip().rstrip('.')))
    out.sort()
    res = []
    for pos, v in out:
        if p['scorer'] == 'mc':
            mm = re.match(r'\s*(?:\\text\{)?\(?([A-J])\b', v)
            ok = bool(mm) and mm.group(1) == p['answer']
        else:
            try:
                g = parse(str(p['answer']), extraction_config=CFG); pr = parse('$' + v + '$', extraction_config=CFG)
                ok = bool(g) and bool(pr) and verify(g, pr)
            except Exception: ok = False
        res.append((pos, v, bool(ok)))
    return res

def ntok(s): return len(tok(s, add_special_tokens=False)['input_ids'])

def tail_stats(think):
    ids = tok(think, add_special_tokens=False)['input_ids']
    tail = ids[-1500:]
    # exact periodicity: smallest period p (<=1200) with tail[-k:]==tail[-k-p:-p] for k=300
    period = None
    for p in range(1, 1200):
        if len(tail) >= 300 + p and tail[-300:] == tail[-300 - p:-p]: period = p; break
    # fraction of tail lines that already appeared earlier in the trace
    lines = [l.strip() for l in think.split('\n') if l.strip()]
    tl = tok.decode(tail)
    tail_lines = [l.strip() for l in tl.split('\n') if l.strip()][1:]
    head_set = collections.Counter(lines[:max(0, len(lines) - len(tail_lines))])
    dup = sum(1 for l in tail_lines if head_set[l] > 0) / max(1, len(tail_lines))
    tb = tl.encode(); cr = len(zlib.compress(tb, 9)) / max(1, len(tb))
    hed = len(HEDGE.findall(tl))
    return {'period': period, 'dup_line_frac': dup, 'zlib_ratio': cr, 'tail_hedges': hed, 'tail_lines': len(tail_lines)}

out = []
for line in open(texts):
    r = json.loads(line); key = (r['id'], r['sample']); row = rows[key]; p = P[r['id']]
    text = r['text']; closed = '</think>' in text
    think = text.split('</think>')[0]
    S = stmts(think, p)
    d = {'id': r['id'], 'sample': r['sample'], 'bench': r['bench'], 'score': row['score'], 'closed': closed,
         'truncated': row['truncated'], 'tokens': row['tokens'], 'think_tokens': ntok(think), 'n_stmts': len(S)}
    if S:
        d['first_stmt_tok'] = ntok(think[:S[0][0]]); d['first_stmt_ok'] = S[0][2]
        d['last_stmt_ok'] = S[-1][2]
        okp = [s for s in S if s[2]]
        d['first_correct_tok'] = ntok(think[:okp[0][0]]) if okp else None
        d['any_correct'] = bool(okp)
        d['distinct_vals'] = len(set(s[1] for s in S))
        # last statement position (where the thinking 'settled')
        d['last_stmt_tok'] = ntok(think[:S[-1][0]])
    if not closed or row['truncated']: d.update(tail_stats(think))
    out.append(d)
json.dump(out, open(outf, 'w'))
print('wrote', len(out))
