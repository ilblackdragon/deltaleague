"""Build a fixed, seeded proxy prompt set from public datasets (no private data)."""
import json, random, sys
from datasets import load_dataset
rng = random.Random(20261105)
out = []
def take(ds, n):
    idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MC_SUFFIX = "\nAnswer with the letter of the correct option, written as \\boxed{X}."
for d in take(load_dataset('HuggingFaceH4/MATH-500', split='test'), int(sys.argv[1])):
    out.append({'bench': 'math500', 'prompt': d['problem'] + MATH_SUFFIX, 'answer': d['answer'], 'scorer': 'math'})
for d in take(load_dataset('openai/gsm8k', 'main', split='test'), int(sys.argv[2])):
    out.append({'bench': 'gsm8k', 'prompt': d['question'] + MATH_SUFFIX, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''), 'scorer': 'math'})
L = 'ABCDEFGHIJ'
for d in take(load_dataset('TIGER-Lab/MMLU-Pro', split='test'), int(sys.argv[3])):
    opts = '\n'.join(f'{L[i]}. {o}' for i, o in enumerate(d['options']))
    out.append({'bench': 'mmlu_pro', 'prompt': f"{d['question']}\n\n{opts}" + MC_SUFFIX, 'answer': d['answer'], 'scorer': 'mc'})
for d in take(load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test'), int(sys.argv[4])):
    opts = '\n'.join(f'{l}. {t}' for l, t in zip(d['choices']['label'], d['choices']['text']))
    out.append({'bench': 'arc_c', 'prompt': f"{d['question']}\n\n{opts}" + MC_SUFFIX, 'answer': d['answerKey'], 'scorer': 'mc'})
for i, o in enumerate(out): o['id'] = i
json.dump(out, open(sys.argv[5], 'w'))
print(len(out), 'prompts')
