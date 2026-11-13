"""Token statistics of thinking traces: truncated vs closed generations. usage: tokstats.py RESULT_PREFIX"""
import json, sys, collections, re
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/user/delta/exp-base/model')
pre=sys.argv[1]
T=[json.loads(l) for l in open(pre+'.texts.jsonl')]
R={(r['id'],r['sample']):r for r in json.load(open(pre+'.json'))['rows']}
def linestart_words(ids):
    """yield (pos, token) for first non-space token of each line (line start = after a token containing newline)"""
    prev_nl=True
    for k,i in enumerate(ids):
        s=tok.convert_ids_to_tokens(i)
        if prev_nl and s.strip('Ġ ĉ')!='' and 'Ċ' not in s: yield k,s; prev_nl=False
        if 'Ċ' in s: prev_nl=True
        elif s.strip('Ġ')!='': prev_nl=False
groups={'closed_ok':collections.Counter(),'closed_bad':collections.Counter(),'trunc_tail':collections.Counter(),'answer':collections.Counter()}
ls={k:collections.Counter() for k in groups}
ntok={k:0 for k in groups}; nls={k:0 for k in groups}
for t in T:
    r=R[(t['id'],t['sample'])]; text=t['text']
    think=text.split('</think>')[0]; ids=tok(think,add_special_tokens=False)['input_ids']
    if r['truncated']: g='trunc_tail'; ids=ids[len(ids)//2:]   # second half = loop region
    elif r['closed']: g='closed_ok' if r['score']>0 else 'closed_bad'
    else: continue
    groups[g].update(tok.convert_ids_to_tokens(ids)); ntok[g]+=len(ids)
    for _,s in linestart_words(ids): ls[g][s]+=1; nls[g]+=1
    if r['closed']:
        a=tok(text.split('</think>')[-1],add_special_tokens=False)['input_ids']
        groups['answer'].update(tok.convert_ids_to_tokens(a)); ntok['answer']+=len(a)
        for _,s in linestart_words(a): ls['answer'][s]+=1; nls['answer']+=1
print('tokens',ntok,'linestarts',nls)
def rate(c,n,s): return 1000*c[s]/max(1,n)
# all tokens: over-representation in trunc tail vs closed_ok thinking
cand=set()
for s,_ in (groups['trunc_tail']+groups['closed_ok']).most_common(400): cand.add(s)
rows=[]
for s in cand:
    a=rate(groups['trunc_tail'],ntok['trunc_tail'],s); b=rate(groups['closed_ok'],ntok['closed_ok'],s)
    rows.append((s,a,b,(a+0.05)/(b+0.05),rate(groups['answer'],ntok['answer'],s)))
out={'ntok':ntok,'nls':nls}
print('\n== all tokens per 1k: trunc_tail vs closed_ok thinking (sorted by ratio, word-like only) ==')
W=[r for r in rows if re.fullmatch(r'Ġ?[A-Za-z\']+|\.\.\.|Ġ\.\.\.|\?|Ġ\?',r[0])]
for s,a,b,q,ans in sorted(W,key=lambda r:-r[3])[:45]: print(f'{s!r:18} trunc {a:7.2f} ok {b:7.2f} ratio {q:5.2f} answer {ans:6.2f}')
print('\n== line-initial tokens per 1k line starts ==')
cand=set(s for s,_ in (ls['trunc_tail']+ls['closed_ok']).most_common(150))
L=[]
for s in cand:
    a=1000*ls['trunc_tail'][s]/nls['trunc_tail']; b=1000*ls['closed_ok'][s]/nls['closed_ok']; c=1000*ls['closed_bad'][s]/max(1,nls['closed_bad'])
    L.append((s,a,b,c,(a+0.5)/(b+0.5),1000*ls['answer'][s]/max(1,nls['answer'])))
for s,a,b,c,q,ans in sorted(L,key=lambda r:-r[1])[:70]: print(f'{s!r:18} trunc {a:7.1f} ok {b:7.1f} bad {c:7.1f} ratio {q:5.2f} answer {ans:6.1f}')
out['all']=rows; out['linestart']=L
json.dump(out,open(pre+'.tokstats.json','w'))
print('\n== candidate hedge tokens: per 1k tokens (trunc_tail / closed_ok / closed_bad / answer) ==')
C=['ĠWait','Wait','Ġwait','wait','ĠHmm','Hmm','Ġhmm','ĠMaybe','Maybe','Ġmaybe','ĠActually','Actually','Ġactually','ĠAlternatively','Alternatively','ĠPerhaps','Perhaps','ĠBut','But','Ġbut','ĠLet','Let','ĠSo','So','ĠNo','No','ĠHold','Hold','ĠOkay','Okay','ĠOK','ĠAh','Ah','ĠOh','Oh','ĠHowever','However','ĠIs','Is','ĠOr','Or','ĠIf','If','ĠThis','This','ĠThen','Then','Ġre','Ġagain','Ġcheck','Ġverify','ĠDouble','Ġdouble','ĠCheck','Check','ĠWAIT','WAIT','...','Ġ...','?','Ġ?','ĠRe','Re','ĠAnother','Another','Ġreconsider','ĠStill','Still','ĠYes','Yes','ĠLooking','ĠRecall','ĠRight','Right','ĠAlright','Alright','ĠNope','Ġmistake','ĠOops','ĠHmmm','ĠUm']
for s in C:
    if s not in tok.get_vocab(): continue
    v=[1000*groups[g][s]/max(1,ntok[g]) for g in ['trunc_tail','closed_ok','closed_bad','answer']]
    l=[1000*ls[g][s]/max(1,nls[g]) for g in ['trunc_tail','closed_ok','closed_bad','answer']]
    print(f'{s!r:18}'+' '.join(f'{x:7.2f}' for x in v)+'   | linestart/1k '+' '.join(f'{x:6.1f}' for x in l))
