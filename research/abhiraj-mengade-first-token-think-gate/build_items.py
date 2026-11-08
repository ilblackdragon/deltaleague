"""Build a local item bank from public benchmarks, excluding the public validation indices.

Output: analysis/items.json. The prompt format is our guess at the private format and is the
main threat to external validity: the private prompts are not published.
"""
import json
import random
import sys

from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
val = json.load(open('experiment/validation.json'))


def used(name):
    out = set()
    for sel in val['text'][name]['selection'].values():
        out |= set(sel.get('indices', []))
    return out


MATH_SUFFIX = r' Please reason step by step and put your final answer in \boxed{}.'
CHOICE_SUFFIX = r' Answer with the letter of the correct option in \boxed{}.'
LET = 'ABCDEFGHIJ'
items = {}


def sample(name, ds, fn, exclude=()):
    idx = [i for i in range(len(ds)) if i not in exclude]
    random.Random(20261106).shuffle(idx)
    out = []
    for i in idx:
        it = fn(ds[i])
        if it:
            it.update(id=f'{name}-{i}', benchmark=name, index=i)
            out.append(it)
        if len(out) == N:
            break
    items[name] = out
    print(name, len(out), flush=True)


def gsm(d):
    return {'prompt': d['question'] + MATH_SUFFIX, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''),
            'scorer': 'math'}


def math500(d):
    return {'prompt': d['problem'] + MATH_SUFFIX, 'answer': d['answer'], 'scorer': 'math'}


def arc(d):
    labels = d['choices']['label']
    key = d['answerKey']
    if key in '12345':
        key = LET[labels.index(key)]
    opts = '\n'.join(f'{LET[i]}. {t}' for i, t in enumerate(d['choices']['text']))
    return {'prompt': f"{d['question']}\n{opts}\n{CHOICE_SUFFIX.strip()}", 'answer': key, 'scorer': 'choice'}


def mmlu(d):
    opts = '\n'.join(f'{LET[i]}. {t}' for i, t in enumerate(d['choices']))
    return {'prompt': f"{d['question']}\n{opts}\n{CHOICE_SUFFIX.strip()}", 'answer': LET[d['answer']], 'scorer': 'choice'}


def csqa(d):
    opts = '\n'.join(f"{l}. {t}" for l, t in zip(d['choices']['label'], d['choices']['text']))
    if d['answerKey'] not in 'ABCDE' or not d['answerKey']:
        return None
    return {'prompt': f"{d['question']}\n{opts}\n{CHOICE_SUFFIX.strip()}", 'answer': d['answerKey'], 'scorer': 'choice'}


sample('gsm8k', load_dataset('openai/gsm8k', 'main', split='test'), gsm, used('gsm8k'))
sample('math500', load_dataset('HuggingFaceH4/MATH-500', split='test'), math500, used('math500'))
sample('arc_challenge', load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test'), arc, used('arc_challenge'))
sample('mmlu', load_dataset('cais/mmlu', 'all', split='test'), mmlu)
sample('commonsense_qa', load_dataset('tau/commonsense_qa', split='validation'), csqa)
json.dump(items, open('analysis/items.json', 'w'), indent=0)
print('total', sum(map(len, items.values())))
