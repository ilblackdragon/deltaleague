"""Fixed stress set (public data only): gsm8k, math500, mmlu, bbh, ifeval, humaneval, mbpp + 5 fixed free prompts."""
import json, random
from datasets import load_dataset, get_dataset_config_names
rng = random.Random(777)
def take(ds, n):
    idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
M = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MC = "\nAnswer with the letter of the correct option, written as \\boxed{X}."
out = []
for d in take(load_dataset('openai/gsm8k', 'main', split='test'), 40):
    out.append({'bench': 'gsm8k', 'prompt': d['question'] + M, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''), 'scorer': 'math'})
for d in take(load_dataset('HuggingFaceH4/MATH-500', split='test'), 40):
    out.append({'bench': 'math500', 'prompt': d['problem'] + M, 'answer': d['answer'], 'scorer': 'math'})
for d in take(load_dataset('cais/mmlu', 'all', split='test'), 40):
    opts = '\n'.join(f'{"ABCD"[i]}. {c}' for i, c in enumerate(d['choices']))
    out.append({'bench': 'mmlu', 'prompt': f"{d['question']}\n\n{opts}" + MC, 'answer': "ABCD"[d['answer']], 'scorer': 'mc'})
tasks = [t for t in get_dataset_config_names('SaylorTwift/bbh')]
pool = []
for t in tasks:
    try: ds = load_dataset('SaylorTwift/bbh', t, split='test')
    except Exception: continue
    for d in take(ds, 3): pool.append(d)
rng.shuffle(pool)
for d in pool[:40]:
    out.append({'bench': 'bbh', 'prompt': d['input'] + "\nThink it through, then give only the final answer within \\boxed{}.", 'answer': d['target'], 'scorer': 'exact'})
for d in take(load_dataset('google/IFEval', split='train'), 40):
    out.append({'bench': 'ifeval', 'prompt': d['prompt'], 'scorer': 'ifeval', 'instruction_id_list': d['instruction_id_list'], 'kwargs': d['kwargs']})
for d in take(load_dataset('openai/openai_humaneval', split='test'), 30):
    out.append({'bench': 'humaneval', 'prompt': "Complete the following Python function. Return the full function in a ```python code block.\n\n```python\n" + d['prompt'] + "```",
                'scorer': 'code', 'test': d['test'] + f"\ncheck({d['entry_point']})\n", 'entry_point': d['entry_point'], 'header': d['prompt']})
for d in take(load_dataset('google-research-datasets/mbpp', 'sanitized', split='test'), 30):
    out.append({'bench': 'mbpp', 'prompt': d['prompt'] + "\nYour code should pass these tests:\n" + '\n'.join(d['test_list']) + "\nReturn the solution in a ```python code block.",
                'scorer': 'code', 'test': '\n'.join(d.get('test_imports', []) + d['test_list']) + '\n', 'header': ''})
FIXED = ["What is 2 + 2?", "How many r's are in the word strawberry?", "Is 1001 a prime number? Explain briefly.",
         "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
         "Write a haiku about autumn leaves."]
for f in FIXED: out.append({'bench': 'fixed', 'prompt': f, 'scorer': 'none'})
for i, o in enumerate(out): o['id'] = i
json.dump(out, open('stress.json', 'w')); print(len(out), {b: sum(o['bench'] == b for o in out) for b in set(o['bench'] for o in out)})
