"""Fast dev pass over the generative public-suite prompts (thinking on, private sampler).

Prompts come from a finished official run (runs/pub_leader/reports) so templates are exact.
usage: .venv/bin/python scripts/pubfast_gen.py MODEL OUT.jsonl [--seed 20261104] [--groups bbh,gsm8k,...]
"""
import argparse, json, os
from pathlib import Path

os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
TEXT = ['bbh', 'gsm8k', 'ifeval', 'math500', 'mmlu_pro']
CODE = ['humaneval', 'mbpp']
REF = Path('runs/pub_leader/reports')


def code_prompt(benchmark, doc):
    if benchmark == 'humaneval':
        return 'Complete this Python function. Return the complete function in a Python code block.\n```python\n' + doc['prompt'] + '\n```'
    return ('Write Python code solving this task. Return code in a Python code block.\n' + doc['text']
            + '\nThe code must pass:\n' + '\n'.join(doc['test_list']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=20261104)
    ap.add_argument('--groups', default=','.join(TEXT + CODE))
    a = ap.parse_args()
    groups = a.groups.split(',')
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    items = []
    for g in groups:
        if g in TEXT:
            rep = json.load(open(REF / f'{g}.json'))
            seen = set()
            for task, xs in rep['samples'].items():
                for x in xs:
                    if (task, x['doc_id']) in seen:
                        continue
                    seen.add((task, x['doc_id']))
                    items.append({'group': g, 'key': f'{task}:{x["doc_id"]}', 'prompt': x['arguments'][0][0]})
        else:
            from delta_cli.public_eval.suite import manifest
            for doc in manifest()['code'][g]['docs']:
                p = tok.apply_chat_template([{'role': 'user', 'content': code_prompt(g, doc)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
                items.append({'group': g, 'key': str(doc['task_id']), 'prompt': p})
    llm = LLM(a.model, gpu_memory_utilization=0.82, max_model_len=34816, max_num_seqs=128,
              language_model_only=True, dtype='bfloat16')
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=8192,
                        seed=a.seed, skip_special_tokens=False)
    outs = llm.generate([it['prompt'] for it in items], sp)
    with open(a.out, 'w') as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            f.write(json.dumps({'group': it['group'], 'key': it['key'], 'text': c.text,
                                'ntok': len(c.token_ids), 'finish': c.finish_reason}) + '\n')


if __name__ == '__main__':
    main()
