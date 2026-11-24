"""How constant is the projection of MLP activations (down_proj input) onto their mean?  If c^T a ~ const, then
W_down += -beta * r c^T acts as a bias-free constant steering vector -beta*r at that layer (pseudo-bias)."""
import sys, json, torch
sys.path.insert(0, 'box')
from common import *
tok, m = load_model('nl_model', threads=3)
L = lm(m); T = {json.loads(l)['id']: json.loads(l) for l in open('results/c_new.texts.jsonl')}
P = {p['id']: p for p in json.load(open(sys.argv[1]))} if len(sys.argv) > 1 else None
acts = {i: [] for i in range(24)}
def mk(i):
    def h(mod, inp): acts[i].append(inp[0][0].detach())
    return h
for i, l in enumerate(L.layers): l.mlp.down_proj.register_forward_pre_hook(mk(i))
for pid in [112, 142]:
    x, g0, lab = build(tok, 'Solve the problem.', T[pid]['text'], 1500)
    with torch.no_grad(): L(input_ids=x[None], use_cache=False)
out = {}
for i in range(24):
    A = torch.cat(acts[i])[20:]  # skip the first prompt tokens
    abar = A.mean(0); c = abar / abar.dot(abar); s = A @ c
    out[i] = {'mean': s.mean().item(), 'sd': s.std().item(), 'cv': (s.std() / s.mean()).item(), 'q05': s.quantile(0.05).item(), 'q95': s.quantile(0.95).item()}
    # best single neuron with lowest CV among active ones
    mu = A.mean(0); sd = A.std(0); cv = sd / mu.abs().clamp_min(1e-6); j = torch.where(mu.abs() > 0.05, cv, torch.full_like(cv, 1e9)).argmin()
    out[i]['best_neuron'] = int(j); out[i]['best_neuron_mu'] = mu[j].item(); out[i]['best_neuron_cv'] = cv[j].item()
    print(i, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[i].items()}, flush=True)
json.dump(out, open('results/mlpstats.json', 'w'), indent=0)
