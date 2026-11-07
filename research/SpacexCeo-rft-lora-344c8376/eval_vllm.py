"""Proxy evaluation with vLLM using the competition's generation settings.
usage: eval_vllm.py MODEL_DIR PROMPTS.json OUT.json [samples] [max_tokens]"""
import json, sys, re, time, os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
def main():
    model, pfile, outf = sys.argv[1:4]
    samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    P = json.load(open(pfile))
    import shutil
    for f in ['preprocessor_config.json', 'video_preprocessor_config.json']:
        if not os.path.exists(os.path.join(model, f)): shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hfextra', f), os.path.join(model, f))
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype='bfloat16', max_model_len=max_tokens + 2048, gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')),
              max_num_seqs=256, seed=0, enable_prefix_caching=True, limit_mm_per_prompt={'image': 0, 'video': 0})
    reqs, meta = [], []
    for p in P:
        text = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for s in range(samples):
            reqs.append(text); meta.append((p, s))
    params = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=max_tokens, seed=1000 + 7919 * p['id'] + s)
              for p, s in meta]
    t0 = time.time()
    outs = llm.generate(reqs, params)
    dt = time.time() - t0
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
    HEDGE = re.compile(r'\b(Wait|wait|Hmm|hmm|Maybe|maybe|Actually|actually|Alternatively|alternatively|But wait)\b')
    def score(p, text):
        if '</think>' not in text: return 0.0, False
        ans = text.split('</think>')[-1].strip()
        if p['scorer'] == 'none': return 0.0, True
        if p['scorer'] == 'exact':
            b = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', ans)
            cand = (b[-1] if b else ans.strip().split('\n')[-1]).replace('\\text{', '').replace('}', '').strip().strip('.').strip()
            norm = lambda x: re.sub(r'\s+', ' ', x).strip().casefold()
            return float(norm(cand) == norm(p['answer']) or norm(cand) == norm(p['answer']).strip('()')), True
        if p['scorer'] == 'ifeval':
            from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
            ok = []
            for iid, kw in zip(p['instruction_id_list'], p['kwargs']):
                ins = INSTRUCTION_DICT[iid](iid); ins.build_description(**{k: v for k, v in kw.items() if v is not None})
                a = ins.get_instruction_args()
                if a and 'prompt' in a: ins.build_description(prompt=p['prompt'])
                ok.append(bool(ans) and ins.check_following(ans))
            return float(bool(ok) and all(ok)), True
        if p['scorer'] == 'code':
            import subprocess, tempfile
            blocks = re.findall(r'```(?:python)?\n(.*?)```', ans, re.S)
            if not blocks: return 0.0, True
            code = blocks[-1]
            if p.get('entry_point') and f"def {p['entry_point']}" not in code: code = p['header'] + code
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f: f.write(code + '\n\n' + p['test']); fn = f.name
            try: r = subprocess.run(['python', fn], capture_output=True, timeout=15); return float(r.returncode == 0), True
            except Exception: return 0.0, True
        if p['scorer'] == 'math':
            g = parse(str(p['answer']), extraction_config=CFG); pr = parse(ans, extraction_config=CFG)
            return float(bool(g) and bool(pr) and verify(g, pr)), True
        m = re.findall(r'\\boxed\{\s*(?:\\text\{)?([A-J])', ans) or re.findall(r'(?:answer is|Answer:)\s*\(?([A-J])\b', ans) or re.findall(r'(?:^|\n)\s*\(?([A-J])[.)]?\s*$', ans)
        return float(bool(m) and m[-1] == p['answer']), True
    rows = []
    for (p, s), o in zip(meta, outs):
        c = o.outputs[0]; text = c.text; n = len(c.token_ids)
        sc, closed = score(p, text)
        think = text.split('</think>')[0]
        rows.append({'id': p['id'], 'bench': p['bench'], 'sample': s, 'score': sc, 'closed': closed, 'tokens': n,
                     'truncated': c.finish_reason == 'length', 'hedges': len(HEDGE.findall(think)), 'answer_tail': text[-300:]})
    by = {}
    for r in rows: by.setdefault(r['bench'], []).append(r)
    summ = {b: {'acc': 100 * sum(r['score'] for r in v) / len(v), 'trunc': sum(r['truncated'] for r in v) / len(v),
                'tokens': sum(r['tokens'] for r in v) / len(v), 'hedges': sum(r['hedges'] for r in v) / len(v), 'n': len(v)} for b, v in by.items()}
    allr = rows
    summ['ALL'] = {'acc': 100 * sum(r['score'] for r in allr) / len(allr), 'trunc': sum(r['truncated'] for r in allr) / len(allr),
                   'tokens': sum(r['tokens'] for r in allr) / len(allr), 'hedges': sum(r['hedges'] for r in allr) / len(allr), 'n': len(allr),
                   'acc_closed': 100 * sum(r['score'] for r in allr if r['closed']) / max(1, sum(r['closed'] for r in allr)), 'seconds': dt}
    json.dump({'model': model, 'summary': summ, 'rows': rows}, open(outf, 'w'))
    with open(outf.replace('.json', '.texts.jsonl'), 'w') as f:
        for (p, s), o in zip(meta, outs):
            f.write(json.dumps({'id': p['id'], 'bench': p['bench'], 'sample': s, 'text': o.outputs[0].text}) + '\n')
    print(json.dumps(summ['ALL']))
    for b in by: print(b, json.dumps(summ[b]))


if __name__ == '__main__':
    main()
