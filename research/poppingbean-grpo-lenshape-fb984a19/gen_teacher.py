"""Teacher traces for distillation: Qwen/Qwen3.5-9B (same tokenizer and chat template as the 0.8B student)
answers public TRAIN-split prompts with thinking on, N samples each; every sample is scored with the
platform scorers. Prompt pools and the probe exclusion come from roshaans' grpo.py (#58).

usage: python gen_teacher.py OUT.jsonl [--prompts 4000] [--n 2] [--max-tokens 6144]
                             [--mix math=0.35,choice=0.30,ifeval=0.25,logic=0.10] [--deadline-minutes 240]
"""
import argparse, json, multiprocessing, os, random, time
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--teacher', default='Qwen/Qwen3.5-9B')
ap.add_argument('--prompts', type=int, default=4000); ap.add_argument('--n', type=int, default=2)
ap.add_argument('--max-tokens', type=int, default=6144); ap.add_argument('--chunk', type=int, default=500)
ap.add_argument('--mix', default='math=0.35,choice=0.30,ifeval=0.25,logic=0.10')
ap.add_argument('--deadline-minutes', type=float, default=240); ap.add_argument('--seed', type=int, default=20261112)
a = ap.parse_args()


def main():
    import grpo  # roshaans' pools: public train splits, probe items excluded
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    t0 = time.time()
    shares = {k: float(v) for k, v in (kv.split('=') for kv in a.mix.split(','))}
    skip = grpo.excluded_questions()
    pools = {'math': [x for x in grpo.math_pool() if x['prompt'][:-len(grpo.SUFFIX)] not in skip],
             'choice': grpo.choice_pool(broad=True), 'ifeval': grpo.ifeval_pool(), 'logic': grpo.logic_pool()}
    rng = random.Random(a.seed)
    items = []
    for k, share in shares.items():
        items += [dict(x, id=f'{k}-{i}') for i, x in enumerate(rng.sample(pools[k], min(len(pools[k]), round(a.prompts * share))))]
    rng.shuffle(items)
    done = {json.loads(l)['id'] for l in open(a.out)} if os.path.exists(a.out) else set()
    items = [x for x in items if x['id'] not in done]
    print(f'{len(done)} done, {len(items)} to do; pools { {k: len(v) for k, v in pools.items()} }', flush=True)

    tok = AutoTokenizer.from_pretrained(a.teacher)
    llm = LLM(a.teacher, dtype='bfloat16', max_model_len=a.max_tokens + 2048, seed=0,
              gpu_memory_utilization=float(os.environ.get('GPU_MEM', '0.90')), max_num_seqs=256)
    sp = SamplingParams(n=a.n, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
                        max_tokens=a.max_tokens, seed=a.seed, skip_special_tokens=False)
    scorer = ProcessPoolExecutor(16, mp_context=multiprocessing.get_context('spawn'))
    ntok = 0
    with open(a.out, 'a') as f:
        for c in range(0, len(items), a.chunk):
            if (time.time() - t0) / 60 > a.deadline_minutes:
                print('deadline reached', flush=True); break
            chunk = items[c:c + a.chunk]
            prompts = [tok.apply_chat_template([{'role': 'user', 'content': x['prompt']}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True) for x in chunk]
            ts = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            jobs, meta = [], []
            for x, o in zip(chunk, outs):
                for s in o.outputs:
                    trunc = len(s.token_ids) >= a.max_tokens
                    jobs.append((x, s.text, trunc)); meta.append((len(s.token_ids), trunc)); ntok += len(s.token_ids)
            scores = list(scorer.map(grpo.score_one, jobs, chunksize=16))
            for i, x in enumerate(chunk):
                samples = [{'text': jobs[i * a.n + j][1], 'tokens': meta[i * a.n + j][0], 'truncated': meta[i * a.n + j][1],
                            'score': scores[i * a.n + j]} for j in range(a.n)]
                f.write(json.dumps({**{k: v for k, v in x.items()}, 'samples': samples}) + '\n')
            f.flush()
            acc = sum(scores) / len(scores); tr = sum(m[1] for m in meta) / len(meta)
            print(f'chunk {c // a.chunk}: {len(chunk)} prompts, {time.time() - ts:.0f}s, acc {acc:.3f}, trunc {tr:.3f}, '
                  f'{ntok / (time.time() - t0):.0f} tok/s overall, {(time.time() - t0) / 60:.1f} min', flush=True)
    print('TEACHER-DONE', flush=True)


if __name__ == '__main__':
    main()
