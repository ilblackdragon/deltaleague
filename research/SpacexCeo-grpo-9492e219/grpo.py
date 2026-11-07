"""Minimal on-policy GRPO at the competition sampler (full-parameter, embeddings frozen). Public TRAIN splits only.

  driver (trainer on GPU 1):  python grpo.py train START_MODEL_DIR WORK_DIR [--steps 20] [--prompts 32] [--group 8] [--lr 2e-6]
  generator (spawned per step on GPU 0): python grpo.py gen CKPT_DIR PROMPTS.json OUT.jsonl GROUP SEED

Each step: the generator samples GROUP completions per prompt from the current checkpoint with the server sampler
(T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 8192 new tokens) and scores them with the strict reward
(correct AND closed </think> AND >= 50 thinking tokens AND not truncated). The trainer computes group-normalised
advantages, drops groups with zero variance, and takes ONE AdamW step on  -sum_i A_i * sum_t log p(y_it) / (n_seq * 8192)
(Dr.-GRPO-style constant length normaliser, on-policy so no ratio/clip, no KL). fp32 master weights + bf16 autocast;
gradient checkpointing; lm-head log-probs computed in chunks. The checkpoint is written back as bf16 with the parent's keys."""
import sys, os, json, time, random, subprocess, shutil, math
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
WORK = os.path.dirname(os.path.abspath(__file__))
STRICT_MIN_THINK = 0; MAXTOK = 8192; THINK_END = 248069  # server has no min-thinking rule (9492e219: 10.6% short thinks, 43.6 private)

def gen_main(ckpt, pfile, out, group, seed):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    sys.path.insert(0, os.path.expanduser('~/dl/rft'))
    from gen_rft import score
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    P = json.load(open(pfile)); tok = AutoTokenizer.from_pretrained(ckpt)
    llm = LLM(model=ckpt, dtype='bfloat16', max_model_len=MAXTOK + 2048, gpu_memory_utilization=float(os.environ.get('VLLM_UTIL', '0.88')), max_num_seqs=256, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={'image': 0, 'video': 0})
    reqs, meta = [], []
    for p in P:
        t = tok.apply_chat_template([{'role': 'user', 'content': p['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for s in range(group): reqs.append(t); meta.append((p, s))
    sp = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=MAXTOK, seed=seed + 7919 * p['id'] + s) for p, s in meta]
    t0 = time.time(); outs = llm.generate(reqs, sp)
    with open(out, 'w') as f:
        for (p, s), o in zip(meta, outs):
            c = o.outputs[0]; ids = list(c.token_ids); text = c.text
            closed = THINK_END in ids; trunc = c.finish_reason == 'length'
            think = ids.index(THINK_END) if closed else len(ids)
            sc = score(p, text) if closed else 0.0
            r = float(sc == 1.0 and closed and not trunc and think >= STRICT_MIN_THINK and ids.count(THINK_END) == 1)
            f.write(json.dumps({'id': p['id'], 'src': p['src'], 's': s, 'prompt_ids': list(o.prompt_token_ids), 'ids': ids, 'reward': r,
                                'score': sc, 'closed': closed, 'trunc': trunc, 'think': think}) + '\n')
    print('gen done', len(outs), round(time.time() - t0), 's', flush=True)

def train_main(start, work, steps, n_prompts, group, lr, adv_mode='std', overlong=0, warmup=0, lenpen=0.0):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText
    os.makedirs(work, exist_ok=True); log = open(os.path.join(work, 'train.log'), 'a')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    pool = json.load(open(os.path.join(work, 'pool.json'))); rng = random.Random(int(os.environ.get('GRPO_SEED', '1234')))
    ckpt = os.path.join(work, 'ckpt')
    if not os.path.exists(ckpt): shutil.copytree(start, ckpt)
    keys = list(safe_open(os.path.join(start, 'model.safetensors'), 'pt').keys())
    m = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.float32).cuda()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); m.config.use_cache = False; m.train()
    LM = m.model.language_model
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith('model.language_model.') and 'embed_tokens' not in n
    train = [p for p in m.parameters() if p.requires_grad]
    say('config', json.dumps({'lr': lr, 'adv': adv_mode, 'overlong': overlong, 'warmup': warmup, 'lenpen': lenpen, 'min_think': STRICT_MIN_THINK})); say('trainable params', sum(p.numel() for p in train), 'frozen', sum(p.numel() for p in m.parameters() if not p.requires_grad))
    E = LM.embed_tokens.weight  # tied head, frozen
    for p in m.parameters():
        if not p.requires_grad: p.data = p.data.to(torch.bfloat16)
    opt = torch.optim.AdamW(train, lr=lr, betas=(0.9, 0.99), eps=1e-12, weight_decay=0.0)
    sd_names = dict(m.named_parameters()); sd_names.update(dict(m.named_buffers()))
    ORIG = safe_open(os.path.join(start, 'model.safetensors'), 'pt'); missing = set()
    _dt = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}
    DT = {k: _dt[ORIG.get_slice(k).get_dtype()] for k in keys}
    def save_ckpt():
        out = {}
        for k in keys:
            t = sd_names.get(k)
            if t is None and k == 'lm_head.weight': t = E
            if t is None: out[k] = ORIG.get_tensor(k); missing.add(k); continue   # e.g. mtp.* (not in the HF module): copied unchanged
            out[k] = t.detach().to(DT[k]).contiguous().cpu()
        save_file(out, os.path.join(ckpt, 'model.safetensors.tmp'), metadata={'format': 'pt'})
        os.replace(os.path.join(ckpt, 'model.safetensors.tmp'), os.path.join(ckpt, 'model.safetensors'))
        say('saved ckpt; keys copied unchanged from parent:', len(missing), sorted(missing)[:3])
    def seq_logp(pids, ids):
        x = torch.tensor(pids + ids, device='cuda')[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = LM(input_ids=x, use_cache=False).last_hidden_state[0, len(pids) - 1:-1]
        tgt = x[0, len(pids):]; tot = 0.0
        for a in range(0, len(ids), 1024):
            def chunk(hc, tc):
                z = (hc.to(torch.bfloat16) @ E.T).float()
                return (z.gather(1, tc[:, None])[:, 0] - torch.logsumexp(z, -1)).sum()
            tot = tot + torch.utils.checkpoint.checkpoint(chunk, h[a:a + 1024], tgt[a:a + 1024], use_reentrant=False)
        return tot
    start_step = 0; MP = os.path.join(work, 'master.pt')
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    if os.path.exists(MP):
        st = torch.load(MP, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for n in names: sd_names[n].copy_(st['params'][n])
        opt.load_state_dict(st['opt']); rng.setstate(st['rng']); start_step = st['step']; say('resumed from step', start_step)
    for step in range(start_step + 1, steps + 1):
        t0 = time.time(); batch = rng.sample(pool, n_prompts)
        pf = os.path.join(work, f'prompts_{step}.json'); json.dump(batch, open(pf, 'w'))
        rf = os.path.join(work, f'roll_{step}.jsonl')
        if not os.path.exists(rf):
            torch.cuda.empty_cache()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('GEN_GPU', '0'))
            r = subprocess.run([sys.executable, __file__, 'gen', ckpt, pf, rf + '.tmp', str(group), str(100000 * step + int(os.environ.get('GRPO_SEED', '1234')) - 1234)], env=env,
                               stdout=open(os.path.join(work, f'gen_{step}.log'), 'w'), stderr=subprocess.STDOUT)
            if r.returncode != 0: say('gen failed step', step); sys.exit(1)
            os.replace(rf + '.tmp', rf)
        R = [json.loads(l) for l in open(rf)]; tg = time.time() - t0
        by = {}
        for r in R: by.setdefault(r['id'], []).append(r)
        items = []
        def shaped(r):  # DAPO-style soft overlong penalty on correct finished answers inside the last `overlong` tokens
            x = r['reward']
            if overlong and x > 0: x = x - 0.5 * min(1.0, max(0.0, (len(r['ids']) - (MAXTOK - overlong)) / overlong))
            return x
        def kimi(g, rs):  # Kimi k1.5 in-group length reward: correct get +w*lam, wrong get +w*min(0, lam)
            if not lenpen: return rs
            L = [len(r['ids']) for r in g]; lo, hi = min(L), max(L)
            if hi == lo: return rs
            return [x + lenpen * ((0.5 - (l - lo) / (hi - lo)) if r['reward'] > 0 else min(0.0, 0.5 - (l - lo) / (hi - lo))) for r, x, l in zip(g, rs, L)]
        for g in by.values():
            rs = kimi(g, [shaped(r) for r in g]); mu = sum(rs) / len(rs); sd = (sum((x - mu) ** 2 for x in rs) / len(rs)) ** .5
            if sd < 1e-6: continue
            items += [(r, (x - mu) / (sd + 1e-4) if adv_mode == 'std' else (x - mu)) for r, x in zip(g, rs)]
        stats = {'reward': round(sum(r['reward'] for r in R) / len(R), 4), 'trunc': round(sum(r['trunc'] for r in R) / len(R), 4),
                 'len': round(sum(len(r['ids']) for r in R) / len(R)), 'groups_used': sum(1 for g in by.values() if len(set(r['reward'] for r in g)) > 1),
                 'groups': len(by), 'seqs': len(items)}
        if items:
            for gr in opt.param_groups: gr['lr'] = lr * min(1.0, step / warmup) if warmup else lr
            opt.zero_grad(set_to_none=True); norm = len(items) * MAXTOK
            items.sort(key=lambda it: -len(it[0]['ids']))
            for r, adv in items:
                (-(adv / norm) * seq_logp(r['prompt_ids'], r['ids'])).backward()
            gn = torch.nn.utils.clip_grad_norm_(train, 1.0).item(); opt.step(); opt.zero_grad(set_to_none=True); stats['grad_norm'] = round(gn, 4)
            save_ckpt()
        torch.save({'params': {n: sd_names[n].detach().cpu() for n in names}, 'opt': opt.state_dict(), 'rng': rng.getstate(), 'step': step}, MP + '.tmp')
        os.replace(MP + '.tmp', MP)
        stats.update({'gen_s': round(tg), 'train_s': round(time.time() - t0 - tg)})
        say('STEP', step, json.dumps(stats))
        if step % 5 == 0: shutil.copytree(ckpt, os.path.join(work, f'ckpt_{step}'), dirs_exist_ok=True)
    say('done')

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'gen': gen_main(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    else:
        a = sys.argv
        opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
        train_main(a[2], a[3], opt('--steps', 20), opt('--prompts', 32), opt('--group', 8), opt('--lr', 2e-6), opt('--adv', 'std'), opt('--overlong', 0), opt('--warmup', 0), opt('--lenpen', 0.0))
