"""Rejection-sampling data for self-distillation (public TRAIN splits only; nothing from validation/test).
usage: gen_rft.py MODEL_DIR OUT.jsonl N_SAMPLES [max_tokens] [shard_id n_shards]
Samples with the competition sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on) and scores each sample."""
import json, sys, os, re, random, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
M = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MC = "\nAnswer with the letter of the correct option, written as \\boxed{X}."
def boxed(s):
    i = s.rfind('\\boxed{')
    if i < 0: return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {'{': 1, '}': -1}.get(s[j], 0); j += 1
    return s[i + 7:j - 1]
def build(seed=20261106):
    rng = random.Random(seed); P = []
    def take(ds, n):
        idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
    for d in take(load_dataset('openai/gsm8k', 'main', split='train'), 1400):
        P.append({'src': 'gsm8k_train', 'prompt': d['question'] + M, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''), 'scorer': 'math'})
    math = []
    for c in get_dataset_config_names('EleutherAI/hendrycks_math'):
        for d in load_dataset('EleutherAI/hendrycks_math', c, split='train'):
            if d['level'] in ('Level 1', 'Level 2', 'Level 3', 'Level 4') and boxed(d['solution']): math.append(d)
    for d in take(math, 1600):
        P.append({'src': 'math_train', 'prompt': d['problem'] + M, 'answer': boxed(d['solution']), 'scorer': 'math'})
    for cfg, n in [('ARC-Challenge', 500), ('ARC-Easy', 300)]:
        for d in take(load_dataset('allenai/ai2_arc', cfg, split='train'), n):
            opts = '\n'.join(f'{l}. {t}' for l, t in zip(d['choices']['label'], d['choices']['text']))
            P.append({'src': 'arc_train', 'prompt': f"{d['question']}\n\n{opts}" + MC, 'answer': d['answerKey'], 'scorer': 'mc'})
    for d in take(load_dataset('cais/mmlu', 'auxiliary_train', split='train'), 600):
        d = d.get('train', d)
        opts = '\n'.join(f'{"ABCD"[i]}. {c}' for i, c in enumerate(d['choices']))
        P.append({'src': 'mmlu_aux', 'prompt': f"{d['question']}\n\n{opts}" + MC, 'answer': "ABCD"[d['answer']], 'scorer': 'mc'})
    for i, p in enumerate(P): p['id'] = i
    return P
def build_v3(seed=20261107):
    rng = random.Random(seed); P = []
    def take(ds, n):
        idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
    for split in ('train', 'validation', 'prompt'):                       # MBPP non-test splits only
        for d in load_dataset('google-research-datasets/mbpp', 'full', split=split):
            P.append({'src': 'mbpp_train', 'prompt': d['text'] + "\nYour code should pass these tests:\n" + '\n'.join(d['test_list']) +
                      "\nReturn the solution in a ```python code block.", 'scorer': 'code',
                      'test': (d.get('test_setup_code') or '') + '\n' + '\n'.join(d['test_list']) + '\n'})
    import ast as _ast
    for d in take(load_dataset('argilla/ifeval-like-data', 'filtered', split='train'), 900):
        try:
            ids = d['instruction_id_list'] if isinstance(d['instruction_id_list'], list) else _ast.literal_eval(d['instruction_id_list'])
            kw = d['kwargs'] if isinstance(d['kwargs'], list) else json.loads(d['kwargs'])
        except Exception: continue
        P.append({'src': 'ifeval_like', 'prompt': d['prompt'], 'scorer': 'ifeval', 'instruction_id_list': ids, 'kwargs': kw})
    for i, p in enumerate(P): p['id'] = 100000 + i
    return P
def build_v4(seed=20261108):
    """private-like TRAIN mix: commonsense / science / reading MC, instruction following, code, a little math."""
    rng = random.Random(seed); P = []
    def take(ds, n):
        idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
    L = 'ABCDEFGHIJ'
    def mc(src, q, opts, g):
        return {'src': src, 'prompt': q + '\n\n' + '\n'.join(f'{L[i]}. {o}' for i, o in enumerate(opts)) + MC, 'answer': L[g], 'scorer': 'mc'}
    for d in take(load_dataset('tau/commonsense_qa', split='train'), 1400):
        P.append(mc('csqa_train', d['question'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
    for cfg, n in [('ARC-Easy', 700), ('ARC-Challenge', 500)]:
        for d in take(load_dataset('allenai/ai2_arc', cfg, split='train'), n):
            if d['answerKey'] in d['choices']['label']:
                P.append(mc('arc_train', d['question'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
    for d in take(load_dataset('allenai/winogrande', 'winogrande_xl', split='train'), 1200):
        P.append(mc('wino_train', 'Choose the option that best fills the blank (_):\n' + d['sentence'], [d['option1'], d['option2']], int(d['answer']) - 1))
    for d in take(load_dataset('aps/super_glue', 'boolq', split='train'), 600):
        P.append(mc('boolq_train', f"{d['passage']}\n\nQuestion: {d['question']}?", ['Yes', 'No'], 0 if d['label'] == 1 else 1))
    for d in take(load_dataset('baber/piqa', split='train'), 700):
        P.append(mc('piqa_train', 'Goal: ' + d['goal'], [d['sol1'], d['sol2']], int(d['label'])))
    for d in take(load_dataset('allenai/openbookqa', 'main', split='train'), 600):
        P.append(mc('obqa_train', d['question_stem'], d['choices']['text'], d['choices']['label'].index(d['answerKey'])))
    for d in take(load_dataset('cais/mmlu', 'auxiliary_train', split='train'), 600):
        d = d.get('train', d); P.append(mc('mmlu_aux', d['question'], d['choices'], d['answer']))
    v3 = build_v3()
    P += [dict(p) for p in v3]                           # MBPP non-test + IFEval-like (fixed parser)
    for d in take(load_dataset('openai/gsm8k', 'main', split='train'), 400):
        P.append({'src': 'gsm8k_train', 'prompt': d['question'] + M, 'answer': d['answer'].split('####')[-1].strip().replace(',', ''), 'scorer': 'math'})
    for i, p in enumerate(P): p['id'] = 200000 + i
    return P
def score(p, text):
    if '</think>' not in text: return 0.0
    ans = text.split('</think>')[-1]
    if p['scorer'] == 'code':
        import subprocess, tempfile
        blocks = re.findall(r'```(?:python)?\n(.*?)```', ans, re.S)
        if not blocks: return 0.0
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f: f.write(blocks[-1] + '\n\n' + p['test']); fn = f.name
        try: return float(subprocess.run(['python3', fn], capture_output=True, timeout=15).returncode == 0)
        except Exception: return 0.0
    if p['scorer'] == 'ifeval':
        from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
        ok = []
        try:
            for iid, kw in zip(p['instruction_id_list'], p['kwargs']):
                ins = INSTRUCTION_DICT[iid](iid); ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
                a = ins.get_instruction_args()
                if a and 'prompt' in a: ins.build_description(prompt=p['prompt'])
                ok.append(bool(ans.strip()) and ins.check_following(ans.strip()))
        except Exception: return 0.0
        return float(bool(ok) and all(ok))
    if p['scorer'] == 'math':
        g = parse('$' + str(p['answer']) + '$', extraction_config=CFG); pr = parse(ans, extraction_config=CFG)
        return float(bool(g) and bool(pr) and verify(g, pr))
    m = re.findall(r'\\boxed\{\s*(?:\\text\{)?([A-J])', ans)
    return float(bool(m) and m[-1] == p['answer'])
def main():
    model, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mt = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    shard, ns = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
    seed_off = int(os.environ.get('SEED_OFFSET', '0')); pset = os.environ.get('PSET', 'v1')
    P = [p for p in {'v1': build, 'v3': build_v3, 'v4': build_v4}[pset]() if p['id'] % ns == shard]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype='bfloat16', max_model_len=mt + 2048, gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')),
              max_num_seqs=256, seed=0, enable_prefix_caching=True, limit_mm_per_prompt={'image': 0, 'video': 0})
    reqs, meta = [], []
    for p in P:
        t = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for s in range(n): reqs.append(t); meta.append((p, s, t))
    sp = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=mt, seed=777 + seed_off + 7919 * p['id'] + s) for p, s, _ in meta]
    t0 = time.time(); outs = llm.generate(reqs, sp)
    with open(out, 'w') as f:
        for (p, s, t), o in zip(meta, outs):
            c = o.outputs[0]; text = c.text
            think = text.split('</think>')[0] if '</think>' in text else text
            f.write(json.dumps({'id': p['id'], 'src': p['src'], 'sample': s + seed_off, 'rendered_prompt': t, 'text': text, 'score': score(p, text),
                                'closed': '</think>' in text, 'truncated': c.finish_reason == 'length', 'tokens': len(c.token_ids),
                                'think_tokens': len(tok(think, add_special_tokens=False)['input_ids'])}) + '\n')
    print('done', len(outs), 'gens in', round(time.time() - t0), 's')
if __name__ == '__main__':
    main()
