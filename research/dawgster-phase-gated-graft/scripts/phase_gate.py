"""Phase gate for the ready graft: make the graft neurons' up-projection a linear 'still thinking' probe,
so silu(gate) * up = s_ready * [in think phase]  (SwiGLU as an AND gate).

fit:   python scripts/phase_gate.py fit MODEL OUT.pt      (multi-format traces: pubfast + proxy runs)
apply: python scripts/phase_gate.py apply GRAFT_MODEL FIT_READY.pt PHASE.pt OUTDIR
check: python scripts/phase_gate.py check MODEL PHASE.pt   (held-out format-wise probe stats)"""
import json, sys, glob, random, os, shutil, torch

L = 23
P = 'model.language_model.'


def traces():
    """(prompt_text, generated_text, format) for many formats."""
    from transformers import AutoTokenizer
    out = []
    p2k = {}
    for g in ['bbh', 'gsm8k', 'ifeval', 'math500', 'mmlu_pro']:
        rep = json.load(open(f'runs/pub_leader/reports/{g}.json'))
        for t, xs in rep['samples'].items():
            for x in xs:
                p2k[f'{t}:{x["doc_id"]}'] = x['arguments'][0][0]
    sys.path.insert(0, 'scripts')
    from pubfast_gen import code_prompt
    from delta_cli.public_eval.suite import manifest
    tok = AutoTokenizer.from_pretrained('models/leader')
    for g in ['humaneval', 'mbpp']:
        for d in manifest()['code'][g]['docs']:
            p2k[str(d['task_id'])] = tok.apply_chat_template([{'role': 'user', 'content': code_prompt(g, d)}],
                                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
    for f in glob.glob('runs/pf_leader_s*.jsonl'):
        for l in open(f):
            r = json.loads(l)
            out.append((p2k[r['key']], r['text'], r['group'], f))
    items = {it['id']: it for it in json.load(open('runs/proxy.json'))}
    for l in open('runs/L5/seed1.jsonl'):
        r = json.loads(l)
        p = tok.apply_chat_template([{'role': 'user', 'content': items[r['id']]['prompt']}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        out.append((p, r['text'], 'proxy_' + r['family'], 'L5'))
    return out


def collect(model_dir, max_per_fmt=60, seed=0):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    lm = getattr(m.model, 'language_model', m.model)
    cap = {}
    lm.layers[L].mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__('x', i[0]))
    T = traces()
    random.Random(seed).shuffle(T)
    cnt = {}
    data = []
    for p, text, fmt, src in T:
        if '</think>' not in text or cnt.get(fmt, 0) >= max_per_fmt:
            continue
        cnt[fmt] = cnt.get(fmt, 0) + 1
        pi = tok(p, add_special_tokens=False).input_ids
        th, ans = text.split('</think>', 1)
        ti = tok(th, add_special_tokens=False).input_ids[-1500:]
        ai = tok('</think>' + ans, add_special_tokens=False).input_ids[:1500]
        ids = pi + ti + ai
        with torch.no_grad():
            m(torch.tensor([ids]).cuda())
        x = cap['x'][0].float().cpu()
        a, b = len(pi), len(pi) + len(ti)
        # think positions: sample 40; answer positions (from the </think> token itself onward): sample 40
        tp = random.Random(len(data)).sample(range(a, b), min(40, b - a))
        ap = random.Random(len(data) + 1).sample(range(b, len(ids)), min(40, len(ids) - b))
        data.append((fmt, x[tp], x[ap]))
    return data


def fit(model_dir, out):
    data = collect(model_dir)
    fmts = sorted({d[0] for d in data})
    # hold out every 4th trace per format
    tr = [d for i, d in enumerate(data) if i % 4]
    te = [d for i, d in enumerate(data) if not i % 4]
    X = torch.cat([torch.cat([t, a]) for _, t, a in tr]); Y = torch.cat([torch.cat([torch.ones(len(t)), torch.zeros(len(a))]) for _, t, a in tr])
    mu = X.mean(0); Xc = X - mu
    w = torch.linalg.solve(Xc.T @ Xc + 30 * torch.eye(X.shape[1]), Xc.T @ (Y - Y.mean()))
    b = Y.mean() - mu @ w
    for f in fmts:
        T = torch.cat([t for g, t, a in te if g == f]); A = torch.cat([a for g, t, a in te if g == f])
        st, sa = T @ w + b, A @ w + b
        print(f'{f:14s} think {st.mean():.2f}±{st.std():.2f} (min {st.min():.2f})   answer {sa.mean():.2f}±{sa.std():.2f} (max {sa.max():.2f})')
    torch.save({'w': w, 'b': b}, out)


def apply(graft_dir, fit_ready, phase, outdir):
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    fr = torch.load(fit_ready); ph = torch.load(phase)
    os.makedirs(outdir, exist_ok=True)
    for fn in os.listdir(graft_dir):
        if fn != 'model.safetensors':
            shutil.copy(os.path.join(graft_dir, fn), outdir)
    meta = safe_open(f'{graft_dir}/model.safetensors', 'pt').metadata(); sd = load_file(f'{graft_dir}/model.safetensors')
    U = P + f'layers.{L}.mlp.up_proj.weight'
    u = sd[U].float()
    pv = ph['w'].clone(); pv[fr['cdim']] += ph['b'] / fr['cmean']   # phase(x) = pv . x, bias folded into const dim
    n1, n2 = fr['neurons']
    u[n1] = pv; u[n2] = pv
    sd[U] = u.to(torch.bfloat16)
    save_file(sd, f'{outdir}/model.safetensors', metadata=meta); print('wrote', outdir)


def step(model_dir, fit_ready, phase, outdir, kappa, a=25.0, c1=0.4, c2=0.6):
    """Build from the leader: out = [silu(a(p-c1)) - silu(a(p-c2))] * s / (a(c2-c1)) * kappa * v.
    Soft step in phase p (1=thinking) times the linear ready score s."""
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    kappa, a, c1, c2 = float(kappa), float(a), float(c1), float(c2)
    fr = torch.load(fit_ready); ph = torch.load(phase)
    os.makedirs(outdir, exist_ok=True)
    for fn in os.listdir(model_dir):
        if fn != 'model.safetensors':
            shutil.copy(os.path.join(model_dir, fn), outdir)
    meta = safe_open(f'{model_dir}/model.safetensors', 'pt').metadata(); sd = load_file(f'{model_dir}/model.safetensors')
    G, U, D = (P + f'layers.{L}.mlp.{n}_proj.weight' for n in ('gate', 'up', 'down'))
    g = sd[G].float(); u = sd[U].float(); d = sd[D].float()
    cd, cm = fr['cdim'], fr['cmean']
    one = torch.zeros_like(fr['w']); one[cd] = 1 / cm
    sv = fr['w'].clone(); sv[cd] += fr['b'] / cm
    pv = ph['w'].clone(); pv[cd] += ph['b'] / cm
    n1, n2 = fr['neurons']
    g[n1] = a * (pv - c1 * one); g[n2] = a * (pv - c2 * one)
    u[n1] = sv; u[n2] = sv
    Em = sd[P + 'embed_tokens.weight'].float(); nw = 1 + sd[P + 'norm.weight'].float()
    C = Em.T @ Em; lam = 1e-3 * C.trace() / C.shape[0]
    vp = torch.linalg.solve(C + lam * torch.eye(C.shape[0]), Em[248069]); vp = vp / (Em[248069] @ vp)
    v = vp / nw * fr['rms'] * kappa / (a * (c2 - c1))
    d[:, n1] = v; d[:, n2] = -v
    sd[G], sd[U], sd[D] = g.to(torch.bfloat16), u.to(torch.bfloat16), d.to(torch.bfloat16)
    save_file(sd, f'{outdir}/model.safetensors', metadata=meta); print('wrote', outdir, 'kappa', kappa)


if __name__ == '__main__':
    {'fit': fit, 'apply': apply, 'step': step}[sys.argv[1]](*sys.argv[2:])
