"""LoRA self-distillation on selected traces. Loss on completion tokens only (thinking + answer + <|im_end|>).
usage: train_lora.py BASE_DIR DATA.jsonl OUT_DIR [--r 16] [--alpha 32] [--lr 1e-4] [--epochs 1] [--tok_per_step 32768]"""
import json, sys, math, random, time, os, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
def arg(k, d, t=float):
    return t(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
base, data, out = sys.argv[1:4]
r, alpha, lr, epochs, tps = arg('--r', 16, int), arg('--alpha', 32.0), arg('--lr', 1e-4), arg('--epochs', 1, int), arg('--tok_per_step', 32768, int)
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(base)
DEV = os.environ.get('DEVICE', 'cuda')
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16).to(DEV)
model.config.use_cache = False
targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith('model.language_model.layers.')
           and n.split('.')[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'out_proj', 'gate_proj', 'up_proj', 'down_proj')]
print('lora targets', len(targets))
model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias='none'))
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads()
END = '<|im_end|>'
base_m = model.base_model.model                     # Qwen3_5ForConditionalGeneration under PEFT
inner = base_m.model.language_model                 # text decoder -> last_hidden_state (final norm applied)
W_head = base_m.lm_head.weight                      # tied with embed_tokens; frozen
def chunk_ce(hc, tc):
    logits = (hc @ W_head.t()).float()
    return torch.nn.functional.cross_entropy(logits, tc, reduction='sum')
ex = []
for l in open(data):
    d = json.loads(l)
    p = tok(d['prompt'], add_special_tokens=False)['input_ids']; c = tok(d['completion'] + END, add_special_tokens=False)['input_ids']
    ex.append((p, c))
print('examples', len(ex), 'completion tokens', sum(len(c) for _, c in ex))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
steps_total = math.ceil(epochs * sum(len(p) + len(c) for p, c in ex) / tps); step = 0; t0 = time.time()
sched = lambda s: lr * min(1.0, (s + 1) / 10) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total)))))
model.train()
for ep in range(epochs):
    random.shuffle(ex); buf, ntok, acc_loss, acc_n = [], 0, 0.0, 0
    for k, (p, c) in enumerate(ex):
        buf.append((p, c)); ntok += len(p) + len(c)
        if ntok < tps and k != len(ex) - 1: continue
        tot = sum(len(c) for _, c in buf)
        for p_, c_ in buf:   # one sequence per forward (variable length, no padding); loss weighted by completion tokens
            ids = torch.tensor([p_ + c_], device=DEV)
            h = inner(input_ids=ids).last_hidden_state[0]              # [T, d], no full-vocab logits materialised
            hs = h[len(p_) - 1: len(p_) + len(c_) - 1]                 # positions predicting completion tokens
            tgt = ids[0, len(p_): len(p_) + len(c_)]
            loss = 0.0
            for a in range(0, hs.shape[0], 1024):                     # chunked CE, logits recomputed in backward
                loss = loss + torch.utils.checkpoint.checkpoint(chunk_ce, hs[a:a + 1024], tgt[a:a + 1024], use_reentrant=False)
            loss = loss / tot
            loss.backward(); acc_loss += loss.item() * tot; acc_n += len(c_)
        for g in opt.param_groups: g['lr'] = sched(step)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1; buf, ntok = [], 0
        if step % 5 == 0: print(f'ep {ep} step {step}/{steps_total} loss {acc_loss / max(1, acc_n):.4f} lr {sched(step):.2e} {time.time() - t0:.0f}s', flush=True); acc_loss, acc_n = 0.0, 0
os.makedirs(out, exist_ok=True)
model.save_pretrained(out)
json.dump({'r': r, 'alpha': alpha, 'lr': lr, 'epochs': epochs, 'tok_per_step': tps, 'targets': targets, 'examples': len(ex)}, open(out + '/train_cfg.json', 'w'))
print('saved', out, round(time.time() - t0), 's')
