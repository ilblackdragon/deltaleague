"""Circuit graft: 'ready to answer' switch -> </think> logit, compiled into 2 repurposed neurons of the last MLP.
Step A (fit): python graft_ready.py fit MODEL "RUN_GLOB" OUT.pt
Step B (build): python graft_ready.py build MODEL FIT.pt OUTDIR KAPPA   (KAPPA = logit change per unit probe score)"""
import json, sys, glob, random, os, shutil, torch
THINK = 248069; L = 23
P = "model.language_model."
def fit(model_dir, run_glob, out):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    lm = getattr(m.model, "language_model", m.model); mlp = lm.layers[L].mlp
    items = {it["id"]: it for it in json.load(open("runs/proxy.json"))}
    rows = [json.loads(l) for f in glob.glob(run_glob) for l in open(f)]
    mode = os.environ.get("MODE", "ready")
    if mode == "decision":   # good close (single close, correct) vs premature close (closed, then truncated)
        good = [r for r in rows if r["text"].count("</think>") == 1 and r["score"]]
        bad = [r for r in rows if r["closed"] and r["trunc"]]
        random.Random(0).shuffle(good); rows = bad + good[:max(250, len(bad))]
    else:
        rows = [r for r in rows if r["text"].count("</think>") == 1]
        random.Random(0).shuffle(rows); rows = rows[:360]
    cap = {}
    mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("x", i[0]))
    mlp.down_proj.register_forward_hook(lambda mod, i, o: cap.__setitem__("a", i[0]))
    lm.norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("rms_in", i[0]))
    X, Y, imp, rms = [], [], torch.zeros(mlp.down_proj.weight.shape[1]), []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": items[r["id"]]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(p, add_special_tokens=False).input_ids
        ti = tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids[:6000]
        if len(ti) < 300: continue
        with torch.no_grad(): m(torch.tensor([pi + ti]).cuda())
        x = cap["x"][0, len(pi):].float().cpu(); a = cap["a"][0, len(pi):].float().cpu()
        imp += a.abs().mean(0); rms.append(cap["rms_in"][0, len(pi):].float().pow(2).mean(-1).sqrt().cpu())
        n = len(ti)
        if mode == "decision":
            last = list(range(n - 6, n))      # positions right before the close
            X.append(x[last]); Y += [0 if r["trunc"] else 1] * len(last)
            if not r["trunc"]:
                neg = random.Random(r["id"]).sample(range(int(.2 * n), int(.7 * n)), 6); X.append(x[neg]); Y += [0] * len(neg)
            continue
        if r["score"]: pos = list(range(n - 150, n)); X.append(x[pos]); Y += [1] * len(pos)
        neg = random.Random(r["id"]).sample(range(int(.2 * n), int(.7 * n)), 60); X.append(x[neg]); Y += [0] * len(neg)
    X = torch.cat(X); Y = torch.tensor(Y, dtype=torch.float); t = 2 * Y - 1
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)); cut = int(.8 * len(X)); tr, te = idx[:cut], idx[cut:]
    mu = X[tr].mean(0); Xc = X - mu
    w = torch.linalg.solve(Xc[tr].T @ Xc[tr] + 10 * torch.eye(X.shape[1]), Xc[tr].T @ t[tr]); b = -(mu @ w)
    s = X[te] @ w + b; acc = ((s > 0).float() == Y[te]).float().mean().item()
    sd = X.std(0); cdim = int((mu.abs() / (sd + 1e-6)).argmax())
    dn = mlp.down_proj.weight.float().cpu().norm(dim=0); imp = imp / len(rows) * dn
    print(f"probe heldout acc {acc:.3f}  s(pos)={s[Y[te]==1].mean():.2f} s(neg)={s[Y[te]==0].mean():.2f}  const dim {cdim} mean {mu[cdim]:.2f} sd {sd[cdim]:.3f}")
    print("least important neurons", imp.argsort()[:4].tolist(), imp.sort().values[:4].tolist())
    torch.save({"w": w, "b": b, "cdim": cdim, "cmean": mu[cdim], "neurons": imp.argsort()[:2].tolist(),
                "rms": torch.cat(rms).median()}, out)
def build(model_dir, fitp, outdir, kappa):
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    f = torch.load(fitp); kappa = float(kappa)
    os.makedirs(outdir, exist_ok=True)
    for fn in os.listdir(model_dir):
        if fn != "model.safetensors": shutil.copy(os.path.join(model_dir, fn), outdir)
    meta = safe_open(f"{model_dir}/model.safetensors", "pt").metadata(); sd = load_file(f"{model_dir}/model.safetensors")
    G, U, D = (P + f"layers.{L}.mlp.{n}_proj.weight" for n in ("gate", "up", "down"))
    g = sd[G].float(); u = sd[U].float(); d = sd[D].float()
    n1, n2 = f["neurons"]
    sv = f["w"].clone(); sv[f["cdim"]] += f["b"] / f["cmean"]          # s(x) = sv . x  (bias folded into const dim)
    one = torch.zeros_like(sv); one[f["cdim"]] = 1 / f["cmean"]          # one . x ~= 1
    g[n1] = sv; g[n2] = -sv; u[n1] = one; u[n2] = one                    # silu(s) - silu(-s) = s
    Em = sd[P + "embed_tokens.weight"].float(); nw = 1 + sd[P + "norm.weight"].float()
    C = Em.T @ Em; lam = 1e-3 * C.trace() / C.shape[0]
    vp = torch.linalg.solve(C + lam * torch.eye(C.shape[0]), Em[THINK])   # targeted push in normed space
    vp = vp / (Em[THINK] @ vp)                                            # exactly +1 on </think> logit
    off = (Em @ vp); off[THINK] = 0
    print("off-target logit per unit: max %.3f  rms %.4f" % (off.abs().max(), off.pow(2).mean().sqrt()))
    v = vp / nw * f["rms"]                                                # back to residual space
    d[:, n1] = kappa * v; d[:, n2] = -kappa * v
    sd[G], sd[U], sd[D] = g.to(torch.bfloat16), u.to(torch.bfloat16), d.to(torch.bfloat16)
    save_file(sd, f"{outdir}/model.safetensors", metadata=meta); print("wrote", outdir, "kappa", kappa, "neurons", n1, n2)
if __name__ == "__main__":
    {"fit": fit, "build": build}[sys.argv[1]](*sys.argv[2:])
