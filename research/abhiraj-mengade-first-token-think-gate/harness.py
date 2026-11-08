"""MLX harness matching the server's decoding: T=1, top_k=20, top_p=.95, presence penalty on generated tokens.

Edits are applied in memory, to the same tensors a submission would change, so a screen does not need
a checkpoint per arm. Scoring mirrors delta_cli/scoring.py.
"""
import json
import os
import re
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

CLOSE = 248069
OPEN = 248068
PENALTY = 1.5


def score_answer(ex, text):
    answer = text.split('</think>')[-1].strip()
    if ex['scorer'] == 'choice':
        boxed = re.findall(r'\\boxed\{([A-E1-5])\}', answer)
        m = boxed or re.findall(r'(?:^|\n)\s*([A-E1-5])[.)]?\s*$', answer)
        return float(bool(m) and m[-1] == str(ex['answer']))
    if ex['scorer'] == 'math':
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        try:
            gold = parse(str(ex['answer']), extraction_config=cfg)
            pred = parse(answer[-4000:], extraction_config=cfg)
            return float(bool(gold) and bool(pred) and verify(gold, pred))
        except Exception:
            return 0.0
    raise ValueError(ex['scorer'])


def load_model(path='experiment/model'):
    # Hard limits: this is a 16 GB laptop with little free disk, so swap must stay out of the picture.
    mx.set_memory_limit(7 * 1024**3)
    mx.set_cache_limit(1024**3)
    model, tok = load(path)
    return model, tok


def params(model):
    """The two tensors the row-level edits touch, in MLX naming."""
    lm = model.language_model.model
    return lm.embed_tokens, lm.norm


def presence(plen):
    def proc(tokens, logits):
        seen = sorted(set(tokens[plen:].tolist())) if tokens.size > plen else []
        if seen:
            logits[:, mx.array(seen)] -= PENALTY
        return logits
    return proc


def render(tok, prompt, thinking=True):
    s = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=thinking)
    return tok.encode(s, add_special_tokens=False)


def run(model, tok, items, thinking=True, max_tokens=8192, seed=1000, batch=16, log=None, prefix=None):
    """Continuous-batched generation. Returns one row per item; `log` (path) gets rows as they finish."""
    mx.random.seed(seed)
    if log and os.path.exists(log):  # resume: keep finished rows
        have = {json.loads(l)['id']: json.loads(l) for l in open(log)}
    else:
        have = {}
    all_items, items = items, [it for it in items if it['id'] not in have]
    prompts = [render(tok, it['prompt'], thinking) + (prefix or []) for it in items]
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=20)
    eos = list(tok.eos_token_ids)
    gen = BatchGenerator(model, stop_tokens=[[t] for t in eos], sampler=sampler, completion_batch_size=batch,
                         prefill_batch_size=min(batch, 4), max_tokens=max_tokens)
    mx.set_wired_limit(6 * 1024**3)  # BatchGenerator raises this to the recommended maximum; undo that.
    uids = gen.insert(prompts, [max_tokens] * len(prompts), logits_processors=[[presence(len(p))] for p in prompts])
    idx = {u: i for i, u in enumerate(uids)}
    out = {u: [] for u in uids}
    rows = [None] * len(items)
    t0 = time.time()
    total = done = 0
    f = open(log, 'a') if log else None
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != 'stop':
                out[r.uid].append(r.token)
            total += 1
            if r.finish_reason is not None:
                i = idx[r.uid]
                ids = out[r.uid]
                text = tok.decode(ids)
                row = dict(id=items[i]['id'], benchmark=items[i]['benchmark'], tokens=len(ids),
                           truncated=r.finish_reason == 'length', closed=CLOSE in ids,
                           close_at=ids.index(CLOSE) if CLOSE in ids else None,
                           score=score_answer(items[i], text), text=text)
                rows[i] = row
                done += 1
                if f:
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                if done % 10 == 0:
                    print(f'  {done}/{len(items)} tok/s={total / (time.time() - t0):.1f} mean={sum(x["score"] for x in rows if x) / done:.3f}', flush=True)
    gen.close()
    new = {r['id']: r for r in rows}
    return [new.get(it['id']) or have[it['id']] for it in all_items]


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r['benchmark'], []).append(r)
    res = {b: round(100 * sum(x['score'] for x in v) / len(v), 1) for b, v in by.items()}
    return dict(all=round(100 * sum(x['score'] for x in rows) / len(rows), 2), by=res,
                trunc=round(sum(x['truncated'] for x in rows) / len(rows), 3),
                closed=round(sum(x['closed'] for x in rows) / len(rows), 3),
                mean_tokens=round(sum(x['tokens'] for x in rows) / len(rows), 1))
