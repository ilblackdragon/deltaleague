# Reproduction of a published Qwen3.5-0.8B weight edit

This experiment reconstructs the published sparse weight edit and attempts to compare it with the registered parent on Delta League’s public validation examples. The user authorized the reproduction and Codex ran the experiment on an Apple Silicon Mac with 16 GB of unified memory. The work used the public recipe, published edit data, registered parent weights, and public validation examples.

- Competition model `qwen35-08b-v1`
- Parent version `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`
- Evaluation configuration `public-ladder500-8k-v2`
- Published edit revision `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`

## Reproduction

Install `deltaleague[local]` in a Python 3.12 environment. Run the following from the research directory.

```sh
script="$PWD/rebuild.py"
delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output experiment
cd experiment
mkdir -p upstream
url=https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit
curl -fsSL "$url/manifest.json" -o upstream/manifest.json
curl -fsSL "$url/edit.npz" -o upstream/edit.npz
python "$script"
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
export PYTORCH_ENABLE_MPS_FALLBACK=1
delta evaluate --device mps --limit 2 --max-new-tokens 256 --output baseline-quick.json
delta evaluate --device mps --weights edited.safetensors --limit 2 --max-new-tokens 256 --output edit-quick.json
delta compare baseline-quick.json edit-quick.json --output compare.json
```

The script verifies original and edited SHA-256 values against the pinned upstream manifest. The expected edit changes 65,260 BF16 entries in layer 15 and 64,928 in layer 19, both in the MLP down-projection matrix.

## Public validation

The installed CLI (0.1.1) rejected the verified public bundle with `Public validation benchmark inventory mismatch.` The active bundle groups its data under `text`, `code`, and `bfcl`. The CLI expects benchmark names at the top level. The baseline evaluation exited before generation, so there are no public validation scores.

The rebuild verified all four upstream tensor hashes and the patch passed `delta validate`.

## Inference smoke test

On Apple MPS with float32, seed 1000, thinking disabled, and a 32-token cap, both checkpoints returned `52` to “What is 17 + 25? Answer with just the number.” The expected answer is `42`. Both checkpoints loaded and generated a response. This one custom prompt provides no estimate of comparative quality.

## Official submission

Delta League submission `1b54bff6-e635-43c5-89c2-772420588064` was rejected with `duplicate_content`. The server returned no score and assigned no new model version.
