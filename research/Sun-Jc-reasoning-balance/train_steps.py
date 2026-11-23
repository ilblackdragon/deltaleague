"""Fit a trust-regularized low-rank edit on paired synthetic reasoning steps."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear


LAYERS = (16, 20)
PARENT_HASH = '2c94041a49dfc1fbf74c1214487dd10dd72e288f5a5e0b6d8fff6d432bb9b59f'
PAIR_HASH = '189448a834b400d91b2a87e63bc4cd982c259939b7aebcbad43fa24cefbbd493'


def encode(row, tokenizer):
    contexts, targets, alternatives = [], [], []
    for i, context in enumerate(row['contexts']):
        full = [tokenizer(context + ' ' + label) for label in row['answers']]
        assert full[0][:-1] == full[1][:-1] and full[0][-1] != full[1][-1]
        contexts.append(full[i][:-1])
        targets.append(full[i][-1])
        alternatives.append(full[1-i][-1])
    # Right padding keeps scored positions causal and enables the differentiable
    # Metal attention kernel for short prompts.
    width = max(65, *(len(seq) for seq in contexts))
    ids = mx.array([seq + [0]*(width-len(seq)) for seq in contexts])
    return ids, mx.array([len(seq)-1 for seq in contexts]), mx.array(targets), mx.array(alternatives)


def logprobs(model, ids, positions):
    backbone = model.language_model.model
    hidden = backbone(ids)[mx.arange(ids.shape[0]), positions]
    logits = backbone.embed_tokens.as_linear(hidden)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def margins(logs, targets, alternatives):
    rows = mx.arange(logs.shape[0])
    return logs[rows, targets] - logs[rows, alternatives]


def loss(model, ids, positions, targets, alternatives, reference):
    logs = logprobs(model, ids, positions)
    discrimination = mx.logaddexp(mx.array(0.0), -margins(logs, targets, alternatives))
    divergence = (mx.exp(reference) * (reference-logs)).sum(axis=-1)
    return (discrimination + 0.5*divergence).mean()


def diagnose(model, examples):
    changes, divergences = [], []
    for ids, positions, targets, alternatives, reference in examples:
        logs = logprobs(model, ids, positions)
        changes.extend(margins(logs-reference, targets, alternatives).tolist())
        divergences.extend((mx.exp(reference)*(reference-logs)).sum(axis=-1).tolist())
    assert all(map(math.isfinite, changes+divergences))
    return dict(contexts=len(changes), mean_relative_margin=statistics.mean(changes),
                improved_margins=sum(v>0 for v in changes), mean_forward_kl=statistics.mean(divergences))


def train(out, plan_path):
    output = out/'factors.safetensors'
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    adapter = Path(plan['adapter'])
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == plan['adapter_sha256']
    spec = importlib.util.spec_from_file_location('pinned_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.Backend(Path(plan['parent']), 'parent')
    assert backend.content_hash == plan['parent_content_hash'] == PARENT_HASH
    mx.random.seed(20261123)
    model = backend.model
    model.freeze()
    for i, layer in enumerate(model.layers):
        if i >= min(LAYERS) and layer.is_linear:
            layer.linear_attn.train()
        if i in LAYERS:
            layer.mlp.down_proj = LoRALinear.from_base(layer.mlp.down_proj, r=8, scale=2.0)
    params = tree_flatten(model.trainable_parameters())
    assert len(params)==4 and all(k.endswith(('lora_a', 'lora_b')) for k, _ in params)
    data_path = out/'pairs.json'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == PAIR_HASH
    data = json.loads(data_path.read_text())
    examples = {}
    started = time.monotonic()
    for split, rows in data.items():
        examples[split] = []
        for row in rows:
            inputs = encode(row, backend.encode)
            ref = mx.stop_gradient(logprobs(model, *inputs[:2]))
            mx.eval(ref)
            examples[split].append((*inputs, ref))
    assert len(examples['train'])==96 and len(examples['diagnostic'])==24
    initial = diagnose(model, examples['diagnostic'][:2])
    assert abs(initial['mean_relative_margin']) < 1e-5 and abs(initial['mean_forward_kl']) < 1e-5
    optimizer = optim.Adam(learning_rate=4e-5)
    step_grad = nn.value_and_grad(model, loss)
    trace = []
    for i, example in enumerate(examples['train'], 1):
        value, grads = step_grad(model, *example)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state, value)
        assert math.isfinite(value.item())
        if i % 16 == 0:
            trace.append(dict(step=i, loss=value.item()))
            print(trace[-1], flush=True)
    diagnostic = diagnose(model, examples['diagnostic'])
    factors = {}
    for i in LAYERS:
        projection = model.layers[i].mlp.down_proj
        factors[f'l{i}.a'] = mx.contiguous(projection.lora_a.T)
        factors[f'l{i}.b'] = mx.contiguous(projection.lora_b.T)
    mx.save_safetensors(str(output), factors)
    report = dict(parent_content_hash=PARENT_HASH, dataset_sha256=PAIR_HASH, layers=LAYERS, rank=8, alpha=16,
                  train_pairs=96, trainable_parameters=sum(p.size for _, p in params),
                  initial=initial, diagnostic=diagnostic, trace=trace, elapsed_seconds=time.monotonic()-started)
    (out/'training.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('baseline_plan', type=Path)
    args = parser.parse_args()
    train(args.out, args.baseline_plan)
