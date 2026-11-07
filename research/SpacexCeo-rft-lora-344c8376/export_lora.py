"""Turn a trained PEFT adapter into (1) a Delta League LoRA submission dir (manifest.json + weights.safetensors) against a parent
version and (2) a full merged model dir for local evaluation, reconstructed with the platform's own reconstruct() formula.
usage: export_lora.py PARENT_MODEL_DIR ADAPTER_DIR SUB_OUT MODEL_OUT PARENT_VERSION_ID PARENT_CONTENT_HASH"""
import sys, json, os, shutil, hashlib, torch
from safetensors.torch import load_file, save_file
par, ad, sub, mout, pid, phash = sys.argv[1:7]
cfg = json.load(open(os.path.join(ad, 'adapter_config.json'))); alpha = float(cfg['lora_alpha'])
w = load_file(os.path.join(ad, 'adapter_model.safetensors'))
tensors, mapping, k = {}, {}, 0
for name in sorted(w):
    if '.lora_A.' not in name: continue
    tgt = name.replace('base_model.model.', '').replace('.lora_A.weight', '.weight').replace('.lora_A.default.weight', '.weight')
    A = w[name].float().contiguous(); B = w[name.replace('lora_A', 'lora_B')].float().contiguous()
    tensors[f'l{k}.a'], tensors[f'l{k}.b'] = A, B
    mapping[tgt] = {'a': f'l{k}.a', 'b': f'l{k}.b', 'alpha': alpha}; k += 1
os.makedirs(sub, exist_ok=True)
save_file(tensors, os.path.join(sub, 'weights.safetensors'))
h = hashlib.sha256(open(os.path.join(sub, 'weights.safetensors'), 'rb').read()).hexdigest()
man = {'schema_version': 'delta.submission.v1', 'format': 'lora', 'operation': 'replace', 'reconstruction_version': 'delta-reconstruct-v1',
       'parent_version_id': pid, 'parent_content_hash': phash, 'model_config_id': 'qwen35-08b-v1', 'tensors': mapping,
       'artifacts': [{'name': 'weights.safetensors', 'sha256': h, 'size_bytes': os.path.getsize(os.path.join(sub, 'weights.safetensors'))}]}
json.dump(man, open(os.path.join(sub, 'manifest.json'), 'w'), indent=2)
from delta_cli.artifacts import reconstruct
res = reconstruct(os.path.join(par, 'model.safetensors'), os.path.join(sub, 'weights.safetensors'), man, allowed_prefix='model.language_model.')
os.makedirs(mout, exist_ok=True)
for f in os.listdir(par):
    if f != 'model.safetensors': shutil.copy(os.path.join(par, f), os.path.join(mout, f))
save_file(res, os.path.join(mout, 'model.safetensors'), metadata={'format': 'pt'})
print('lora tensors', k, 'targets', len(mapping), 'alpha', alpha, 'patch sha', h, 'size', os.path.getsize(os.path.join(sub, 'weights.safetensors')))
