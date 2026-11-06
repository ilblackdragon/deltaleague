"""One bounded teacher-correction experiment; public ARC letter-choice proxy only.

No hidden data, network inference, parameter search, or automatic submission.
Run with --workspace /absolute/path/to/experiment after competition init.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import time

os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TEACHER = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TARGET = 'model.language_model.layers.23.mlp.down_proj.weight'
SEED = 42
RANK = 4
STEPS = 40
LR = 0.001
MAX_RELATIVE_NORM = 0.005


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def public_data(w):
    from datasets import load_dataset
    manifest = json.loads((w / 'validation.json').read_text())
    evaluation = json.loads((w / 'evaluation.json').read_text())
    assert digest(w / 'validation.json') == evaluation['validation_sha256']
    assert manifest['scope'] == 'public-validation-only'
    selection = manifest['text']['arc_challenge']['selection']['arc_challenge']
    revision = manifest['dataset_revisions']['allenai/ai2_arc']
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test',
                      revision=revision, cache_dir=str(w / 'data-cache'))
    rows = []
    for idx, expected in zip(selection['indices'], selection['doc_sha256'], strict=True):
        doc = ds[idx]
        actual = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        assert actual == expected, f'Public document hash mismatch at {idx}'
        choices = doc['choices']['text']
        assert 2 <= len(choices) <= 5
        prompt = 'Answer the science question. Reply with only the letter of the correct choice.\n\n'
        prompt += doc['question'] + '\n'
        prompt += '\n'.join(f'{chr(65+i)}. {text}' for i, text in enumerate(choices))
        prompt += '\n\nAnswer:'
        rows.append({'id': doc['id'], 'dataset_index': idx, 'doc_sha256': actual,
                     'prompt': prompt, 'n_choices': len(choices),
                     'label': doc['choices']['label'].index(doc['answerKey'])})
    assert len(rows) == 64
    indices = list(range(len(rows)))
    random.Random(SEED).shuffle(indices)
    dev, holdout = sorted(indices[:32]), sorted(indices[32:])
    return rows, dev, holdout, revision


def load_model(path, student=False):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    cls = AutoModelForImageTextToText if student else AutoModelForCausalLM
    model, info = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float32, use_safetensors=True,
                                     output_loading_info=True, attn_implementation='sdpa')
    assert not any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
    model.requires_grad_(False).eval()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tok


def collect(model, tok, rows, indices, student=False, phase=None):
    """One forward per example, zero generated tokens, no sampling."""
    ids = [tok.encode(chr(65+i), add_special_tokens=False) for i in range(5)]
    assert all(len(x) == 1 for x in ids)
    ids = [x[0] for x in ids]
    logits, features, residuals = [], [], []
    captured = {}
    handles = []
    if student:
        down = model.model.language_model.layers[-1].mlp.down_proj
        norm = model.model.language_model.norm
        def capture_x(module, inputs):
            captured['x'] = inputs[0][:, -1, :].detach().clone()
        def capture_h(module, inputs):
            captured['h'] = inputs[0][:, -1, :].detach().clone()
        handles = [down.register_forward_pre_hook(capture_x), norm.register_forward_pre_hook(capture_h)]
    start = time.monotonic()
    with torch.no_grad():
        for n, idx in enumerate(indices):
            row = rows[idx]
            rendered = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(rendered, return_tensors='pt', add_special_tokens=False)
            assert inputs.input_ids.shape[1] <= 512, 'Fixed input-token cap exceeded'
            out = model(**inputs, use_cache=False, logits_to_keep=1)
            scores = out.logits[0, -1, ids].float().clone()
            scores[row['n_choices']:] = -1e9
            logits.append(scores)
            if student:
                features.append(captured['x'][0])
                residuals.append(captured['h'][0])
            name = phase or ('student' if student else 'teacher')
            print(f"{name} {n+1}/{len(indices)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    for handle in handles:
        handle.remove()
    result = {'logits': torch.stack(logits)}
    if student:
        result.update(x=torch.stack(features), h=torch.stack(residuals),
                      norm_weight=norm.weight.detach().clone(),
                      head=model.get_output_embeddings().weight[ids].detach().clone())
    return result


def proxy_logits(cache, delta, counts):
    h = cache['h'] + F.linear(cache['x'], delta)
    z = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
    z = z * (1 + cache['norm_weight'])
    scores = F.linear(z, cache['head'])
    mask = torch.arange(5)[None, :] >= counts[:, None]
    return scores.masked_fill(mask, -1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--stage', choices=['student', 'teacher', 'edit'], required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    resource.setrlimit(resource.RLIMIT_AS, (64 * 1024**3, 64 * 1024**3))
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    rows, dev, holdout, revision = public_data(w)
    results = w / 'research-local'
    results.mkdir(exist_ok=True)
    start = time.monotonic()
    if args.stage == 'student':
        assert not (results / 'student.safetensors').exists()
        model, tok = load_model(w / 'model', student=True)
        cache = collect(model, tok, rows, list(range(64)), student=True)
        save_file(cache, results / 'student.safetensors')
    elif args.stage == 'teacher':
        assert not (results / 'teacher.safetensors').exists()
        model, tok = load_model(w / 'teacher')
        cache = collect(model, tok, rows, dev)
        save_file(cache, results / 'teacher.safetensors')
    else:
        assert not (w / 'edited.safetensors').exists(), 'Refuse candidate overwrite or second trial'
        cache = load_file(results / 'student.safetensors')
        teacher = load_file(results / 'teacher.safetensors')['logits']
        labels = torch.tensor([r['label'] for r in rows])
        counts = torch.tensor([r['n_choices'] for r in rows])
        before = cache['logits'].argmax(-1)
        correction = [idx for j, idx in enumerate(dev) if teacher[j].argmax() == labels[idx] and before[idx] != labels[idx]]
        print('Teacher-correct/student-wrong development examples:', len(correction), flush=True)
        assert correction, 'No verified teacher corrections; stop without a fabricated edit'
        base = load_file(w / 'model/model.safetensors')
        original = base[TARGET].float()
        a = torch.nn.Parameter(torch.randn(RANK, original.shape[1]) * 0.01)
        b = torch.nn.Parameter(torch.zeros(original.shape[0], RANK))
        opt = torch.optim.AdamW([a, b], lr=LR, weight_decay=0)
        # Cached forward must reproduce original letter logits before any training.
        zero = proxy_logits(cache, torch.zeros_like(original), counts)
        assert torch.allclose(zero, cache['logits'], atol=2e-4, rtol=2e-5)
        retention = torch.softmax(cache['logits'][dev], dim=-1)
        losses = []
        for step in range(STEPS):
            opt.zero_grad()
            # Holdout activations never participate in the loss or optimizer.
            sub = {k: v[dev] if k in ['h', 'x'] else v for k, v in cache.items()}
            scores = proxy_logits(sub, b @ a, counts[dev])
            correction_positions = [dev.index(idx) for idx in correction]
            ce = F.cross_entropy(scores[correction_positions], labels[correction])
            kl = F.kl_div(F.log_softmax(scores, dim=-1), retention, reduction='batchmean')
            loss = ce + 0.5 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_([a, b], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        raw_delta = (b @ a).detach()
        scale = min(1.0, MAX_RELATIVE_NORM * original.norm().item() / raw_delta.norm().item())
        base[TARGET] = (original + raw_delta * scale).to(base[TARGET].dtype).contiguous()
        actual_delta = base[TARGET].float() - original
        assert torch.isfinite(base[TARGET]).all() and torch.count_nonzero(actual_delta) > 0
        save_file(base, w / 'edited.safetensors')
        cached_after = proxy_logits(cache, actual_delta, counts)
        # Verify the exported BF16 edit by running the actual model again on all 64.
        del base, original, opt, a, b
        gc.collect()
        model, tok = load_model(w / 'model', student=True)
        with torch.no_grad():
            model.model.language_model.layers[-1].mlp.down_proj.weight.add_(actual_delta)
        candidate = collect(model, tok, rows, list(range(64)), phase='candidate')
        assert torch.allclose(candidate['logits'], cached_after, atol=5e-4, rtol=5e-5)
        after = candidate['logits'].argmax(-1)
        save_file(candidate, results / 'candidate.safetensors')
        def stats(indices):
            old = (before[indices] == labels[indices]).float()
            new = (after[indices] == labels[indices]).float()
            diff = new - old
            gen = torch.Generator().manual_seed(SEED)
            boot = diff[torch.randint(len(indices), (10000, len(indices)), generator=gen)].mean(-1) * 100
            wins, losses_n = int((diff > 0).sum()), int((diff < 0).sum())
            discordant = wins + losses_n
            p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses_n)+1)) / 2**discordant)
            return {'n': len(indices), 'parent_correct': int(old.sum()), 'candidate_correct': int(new.sum()),
                    'parent_accuracy': float(old.mean()*100), 'candidate_accuracy': float(new.mean()*100),
                    'difference_pp': float(diff.mean()*100), 'wins': wins, 'losses': losses_n,
                    'bootstrap_95_ci_pp': torch.quantile(boot, torch.tensor([.025,.975])).tolist(),
                    'exact_sign_test_p': p}
        report = {'status': 'completed_local_proxy_only', 'official_candidate_scores': None,
            'workspace': json.loads((w/'delta-workspace.json').read_text()),
            'parent_content_hash': json.loads((w/'model/delta-model.json').read_text())['content_hash'],
            'teacher': TEACHER, 'teacher_revision': REVISION, 'dataset_revision': revision,
            'validation_sha256': digest(w/'validation.json'), 'seed': SEED, 'bootstrap_seed': SEED,
            'target': TARGET, 'rank': RANK, 'alpha': RANK, 'steps': STEPS, 'learning_rate': LR,
            'retention_kl_weight': .5, 'gradient_clip': 1., 'raw_delta_scale': scale,
            'relative_delta_norm': float(actual_delta.norm()/load_file(w/'model/model.safetensors')[TARGET].float().norm()),
            'changed_elements': int(torch.count_nonzero(actual_delta)),
            'teacher_development_accuracy': float((teacher.argmax(-1) == labels[dev]).float().mean()*100),
            'correction_ids': [rows[i]['id'] for i in correction],
            'development_ids': [rows[i]['id'] for i in dev], 'holdout_ids': [rows[i]['id'] for i in holdout],
            'development': stats(dev), 'holdout': stats(holdout), 'training_loss_first': losses[0],
            'training_loss_last': losses[-1], 'edited_sha256': digest(w/'edited.safetensors'),
            'parent_weights_sha256': digest(w/'model/model.safetensors'),
            'versions': {name: importlib.metadata.version(name) for name in ['torch','transformers','datasets','safetensors','deltaleague']},
            'compute': {'threads':16, 'dtype':'float32', 'device':'cpu', 'max_input_tokens':512,
                        'generated_tokens':0, 'thinking':False, 'trials':1},
            'limitations': ['Answer-letter proxy differs from pinned native ARC acc/acc_norm and official evaluation.',
                '32-example public holdout is small; development performance is training-contaminated.',
                'Teacher evaluated on development only; only label-verified corrections used.',
                'CLI local evaluator rejects current public selection manifest; official scorer unchanged.']}
        write(results/'results.json', report)
        print(json.dumps({'development':report['development'], 'holdout':report['holdout']}), flush=True)
    write(results/f'{args.stage}-runtime.json', {'seconds':time.monotonic()-start,
          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == '__main__':
    main()
