"""Proxy B: a private-like public mix (held-out eval splits only), weighted toward benchmarks that correlate with the private
score on the leaderboard. usage: build_proxyb.py OUT.json"""
import json, random, sys
from datasets import load_dataset, get_dataset_config_names
rng = random.Random(20261106)
def take(ds, n):
    idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
M = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MC = "\nAnswer with the letter of the correct option, written as \\boxed{X}."
def mc(bench, q, opts, gold_idx):
    L = 'ABCDEFGHIJ'; return {'bench': bench, 'prompt': q + '\n\n' + '\n'.join(f'{L[i]}. {o}' for i, o in enumerate(opts)) + MC, 'answer': L[gold_idx], 'scorer': 'mc'}
out = []
for d in take(load_dataset('tau/commonsense_qa', split='validation'), 50):
    out.append(mc('csqa', d['question'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
for d in take(load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test'), 50):
    out.append(mc('arc_e', d['question'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
for d in take(load_dataset('allenai/winogrande', 'winogrande_xl', split='validation'), 50):
    out.append(mc('winogrande', 'Choose the option that best fills the blank (_):\n' + d['sentence'], [d['option1'], d['option2']], int(d['answer']) - 1))
for d in take(load_dataset('aps/super_glue', 'boolq', split='validation'), 40):
    out.append(mc('boolq', f"{d['passage']}\n\nQuestion: {d['question']}?", ['Yes', 'No'], 0 if d['label'] == 1 else 1))
for d in take(load_dataset('baber/piqa', split='validation'), 40):
    out.append(mc('piqa', 'Goal: ' + d['goal'], [d['sol1'], d['sol2']], int(d['label'])))
for d in take(load_dataset('allenai/openbookqa', 'main', split='test'), 30):
    out.append(mc('obqa', d['question_stem'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
for d in take(load_dataset('cais/mmlu', 'all', split='test'), 40):
    out.append(mc('mmlu', d['question'], d['choices'], d['answer']))
for d in take(load_dataset('google/IFEval', split='train'), 50):
    out.append({'bench': 'ifeval', 'prompt': d['prompt'], 'scorer': 'ifeval', 'instruction_id_list': d['instruction_id_list'], 'kwargs': d['kwargs']})
for d in take(load_dataset('openai/openai_humaneval', split='test'), 40):
    out.append({'bench': 'humaneval', 'prompt': "Complete the following Python function. Return the full function in a ```python code block.\n\n```python\n" + d['prompt'] + "```",
                'scorer': 'code', 'test': d['test'] + f"\ncheck({d['entry_point']})\n", 'entry_point': d['entry_point'], 'header': d['prompt']})
pool = []
for t in get_dataset_config_names('SaylorTwift/bbh'):
    try: pool += take(load_dataset('SaylorTwift/bbh', t, split='test'), 2)
    except Exception: pass
rng.shuffle(pool)
for d in pool[:30]:
    out.append({'bench': 'bbh', 'prompt': d['input'] + "\nThink it through, then give only the final answer within \\boxed{}.", 'answer': d['target'], 'scorer': 'exact'})
for d in take(load_dataset('openai/gsm8k', 'main', split='test'), 20):
    out.append({'bench': 'gsm8k', 'prompt': d['question'] + M, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''), 'scorer': 'math'})
for d in take(load_dataset('HuggingFaceH4/MATH-500', split='test'), 20):
    out.append({'bench': 'math500', 'prompt': d['problem'] + M, 'answer': d['answer'], 'scorer': 'math'})
for i, o in enumerate(out): o['id'] = 50000 + i
json.dump(out, open(sys.argv[1], 'w')); print(len(out), {b: sum(o['bench'] == b for o in out) for b in dict.fromkeys(o['bench'] for o in out)})
