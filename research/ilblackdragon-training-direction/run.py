"""Bounded public reasoning diagnosis and direct weight editing; no optimizer."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import signal
import time

os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from safetensors.torch import load_file, save_file

LAYERS = [15, 19]


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze_data(w):
    # Only training split. Never load the League validation manifest or test split.
    path=w/'local/selected.json'
    if path.exists():
        rows=json.loads(path.read_text())
        plan=json.loads((w/'plan.json').read_text())
        assert sha(path)==plan['selected_sha256']
        return rows
    raise RuntimeError('Run prepare.py once to freeze the disjoint data partitions')


def repetition(ids):
    grams = [tuple(ids[i:i+8]) for i in range(max(0, len(ids)-7))]
    counts = Counter(grams)
    marked = set()
    for i, gram in enumerate(grams):
        if counts[gram] >= 3:
            marked.update(range(i, i+8))
    start = max(0, len(ids)-256)
    tail = [i for i in sorted(marked) if i >= start]
    coverage = len(tail) / max(1, len(ids)-start)
    return {'tail_repetition_coverage': coverage, 'loop_detected': coverage >= .2,
            'loop_positions': tail, 'max_8gram_occurrences': max(counts.values(), default=0)}


def load_model(w, teacher=False, edited=False):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    directory = w/('teacher' if teacher else 'model')
    cls = AutoModelForCausalLM if teacher else AutoModelForImageTextToText
    # Teacher BF16 reduces memory traffic on the local AVX512-BF16 CPU. The
    # parent/candidate comparison and student activation analysis stay float32.
    model, info = cls.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16 if teacher else torch.float32,
        use_safetensors=True, output_loading_info=True, attn_implementation='sdpa')
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), info
    model.requires_grad_(False).eval()
    if edited:
        weights = load_file(w/'edited.safetensors')
        for layer in LAYERS:
            model.model.language_model.layers[layer].mlp.down_proj.weight.copy_(
                weights[f'model.language_model.layers.{layer}.mlp.down_proj.weight'].float())
        del weights
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    return model, tok


def render(tok, prompt):
    return tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)


def generate(model, tok, row, deadline, max_new_tokens=8192):
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, set_seed
    from delta_cli.scoring import score_answer
    inputs = tok(render(tok, row['prompt']), return_tensors='pt', add_special_tokens=False)
    n = inputs.input_ids.shape[1]
    assert n + max_new_tokens <= 34816
    class Presence(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[0, input_ids[0, n:].unique()] -= 1.5
            return scores
    class Watch(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            count = input_ids.shape[1]-n
            if count % 128 == 0:
                print(f"{row['id']} tokens={count} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if time.time() >= deadline:
                raise TimeoutError('Approved experiment wall-clock budget exhausted')
            return False
    set_seed(1000)
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1., top_p=.95, top_k=20,
            logits_processor=LogitsProcessorList([Presence()]),
            stopping_criteria=StoppingCriteriaList([Watch()]))[0, n:]
    ids = output.tolist()
    text = tok.decode(ids, skip_special_tokens=False)
    for special in [tok.eos_token, tok.pad_token]:
        if special:
            text = text.replace(special, '')
    closed = '</think>' in text
    score = score_answer(row, text) if closed else 0.
    result = {k:row[k] for k in ['id','dataset_index','split']}
    result.update(text=text, token_ids=ids, tokens=len(ids), truncated=len(ids)==max_new_tokens,
                  max_new_tokens=max_new_tokens,
                  reasoning_closed=closed, score=score, seconds=time.monotonic()-start,
                  **repetition(ids))
    print(json.dumps({k:v for k,v in result.items() if k not in ['text','token_ids','loop_positions']}), flush=True)
    return result


def read_traces(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def capture(model, tok, row, trace, loop):
    """Replay both texts inside the student; never subtract cross-model activations."""
    prompt_ids = tok.encode(render(tok, row['prompt']), add_special_tokens=False)
    text = trace['text'].split('</think>')[0]
    if not loop:
        # Qwen3 emits this marker, whereas Qwen3.5's generation prompt already
        # includes it. Replay reasoning content without a nested opening tag.
        text=text.removeprefix('<think>').lstrip('\n')
    continuation = tok.encode(text, add_special_tokens=False)
    positions = list(range(max(0,len(continuation)-256),len(continuation)))
    assert positions
    positions = torch.tensor([len(prompt_ids)+p for p in positions])
    captured, handles = {}, []
    for layer in LAYERS:
        def hook(module, inputs, output, layer=layer):
            captured[layer] = {'mean_output':output[0,positions].float().mean(0).clone(),
                               'mean_input_square':inputs[0][0,positions].float().square().mean(0).clone()}
        handles.append(model.model.language_model.layers[layer].mlp.down_proj.register_forward_hook(hook))
    with torch.no_grad():
        model(input_ids=torch.tensor([prompt_ids+continuation]), use_cache=False, logits_to_keep=1)
    for handle in handles:
        handle.remove()
    return captured


def edit(w, rows):
    base_traces = {r['id']:r for r in read_traces(w/'local/train_student.generations.jsonl')}
    teacher_traces = read_traces(w/'local/train_teacher.generations.jsonl')
    pairs = [r for r in teacher_traces if r['score']==1 and r['reasoning_closed']
             and not r['truncated'] and base_traces[r['id']]['score']==0]
    assert len(base_traces)==16 and len(teacher_traces)==16, 'Do not select pairs from incomplete coverage'
    assert all(r['split']=='training' for r in list(base_traces.values())+teacher_traces)
    contrast_kind='training_failed_student_vs_completed_correct_teacher'
    if len(pairs)<3:
        dump(w/'local/outcome.json', {'status':'no_supported_edit','eligible_pairs':len(pairs),
             'reason':'Fewer than three complete teacher-correct/student-failed training pairs.'})
        print('Fewer than three supported training pairs; no edit.',flush=True)
        return
    assert not (w/'edited.safetensors').exists(), 'One candidate only'
    model,tok = load_model(w)
    rows=[r for r in rows if r['split']=='training']
    lookup = {r['id']:r for r in rows}
    bad,good = [],[]
    for pair in pairs:
        bad.append(capture(model,tok,lookup[pair['id']],base_traces[pair['id']],True))
        good.append(capture(model,tok,lookup[pair['id']],pair,False))
    del model,tok
    gc.collect()
    weights = load_file(w/'model/model.safetensors')
    changes = []
    for layer in LAYERS:
        name = f'model.language_model.layers.{layer}.mlp.down_proj.weight'
        original = weights[name].float()
        direction = torch.stack([b[layer]['mean_output']-g[layer]['mean_output'] for b,g in zip(bad,good)]).mean(0)
        assert torch.isfinite(direction).all() and direction.norm()>1e-8
        u = direction/direction.norm()
        activity = torch.stack([b[layer]['mean_input_square'] for b in bad]).mean(0).sqrt()
        projection = u @ original
        columns = torch.argsort(projection.abs()*activity,descending=True,stable=True)[:64]
        delta = -.5 * torch.outer(u, projection[columns])
        scale = min(1., .005*original.norm().item()/delta.norm().item())
        changed = original.clone()
        changed[:,columns] += delta*scale
        weights[name] = changed.to(weights[name].dtype).contiguous()
        diff = weights[name].float()-original
        assert torch.isfinite(weights[name]).all()
        changes.append({'tensor':name,'columns':columns.tolist(),'scale':scale,
                        'changed_elements':int(torch.count_nonzero(diff)),
                        'relative_delta_norm':float(diff.norm()/original.norm())})
    assert sum(c['changed_elements'] for c in changes)>0
    save_file(weights,w/'edited.safetensors')
    dump(w/'local/edit.json', {'pairs':[p['id'] for p in pairs], 'contrast_kind':contrast_kind,
        'teacher_full_response_scores_unchanged':True, 'changes':changes,
        'edited_sha256':sha(w/'edited.safetensors'),'gradient_training':False,'candidate_count':1})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--stage',choices=['train_student','train_teacher','edit','heldout_student','heldout_teacher','heldout_candidate'],required=True)
    args=p.parse_args(); w=args.workspace.resolve()
    plan=json.loads((w/'plan.json').read_text())
    assert plan['proposed_budget']['approved']
    identity=json.loads((w/'delta-workspace.json').read_text())
    assert identity['parent_version_id']==plan['parent_version_id']
    assert identity['config_id']==plan['evaluation_id']
    remaining=int(plan['deadline_unix']-time.time())
    assert remaining>0, 'Approved wall-clock budget exhausted'
    signal.alarm(remaining)
    resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    (w/'local').mkdir(exist_ok=True)
    rows=freeze_data(w)
    start=time.monotonic()
    if args.stage=='edit':
        edit(w,rows)
    else:
        split='training' if args.stage.startswith('train_') else 'heldout'
        selected=[r for r in rows if r['split']==split]
        if split=='heldout':
            assert (w/'local/edit.json').exists(), 'Freeze edit before opening heldout outcomes'
        path=w/f'local/{args.stage}.generations.jsonl'
        done=read_traces(path)
        remaining_rows=[r for r in selected if r['id'] not in {d['id'] for d in done}]
        if remaining_rows:
            model,tok=load_model(w,teacher=args.stage.endswith('_teacher'),edited=args.stage=='heldout_candidate')
            with path.open('a') as stream:
                for row in remaining_rows:
                    result=generate(model,tok,row,plan['deadline_unix'],8192)
                    stream.write(json.dumps(result)+'\n');stream.flush()
    dump(w/f'local/{args.stage}-runtime.json',{'seconds':time.monotonic()-start,
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'dtype':'bfloat16' if args.stage.endswith('_teacher') else 'float32',
         'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','deltaleague']}})


if __name__=='__main__':
    main()
