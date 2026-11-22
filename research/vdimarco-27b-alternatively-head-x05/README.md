# Alternatively output-row ×0.5 on the 27B Wait/Hmm leader

Status: built and locally `delta validate`d. No official score yet. No local accuracy evidence. This host has no Blackwell GPU, so no 27B NVFP4 inference was run.

## Hypothesis

Parent: `5065b8a8-e87d-4f06-b52b-183fcb3bab8c` (mihaikoodo, "PR #91 A + Wait/Hmm output rows x0.25"). Evaluation: `qwen38-27b-nvfp4-agentic300-8k-v1`. The parent is the current 27B private-test leader at 30.00% (90/300) versus 22.00% for the original, with public validation 74.71%. Its reported private-test truncation rate is still about 63%.

PR #91 ported the 0.8B anti-overthinking recipe: dampen Wait/Hmm rows, raise `</think>`, and apply a final RMSNorm gain. The parent then further reduced only the Wait/Hmm *output* rows to ×0.25. That helped, but the same seven hesitation tokens cannot explain every long thinking loop. A common leftover discourse marker is "Alternatively" / "alternatively", which often starts a new reasoning branch after the model has already spent budget.

Hypothesis: scaling only the output rows of those three tokens by ×0.5, the original 0.8B hesitation dose, will slightly reduce stall-and-retry loops without touching the parent's already-tuned Wait/Hmm rows, `</think>` row, input embeddings, or final norm. This is untested. The tokens can also appear in ordinary answers, so the change may hurt some items. ×0.5 was chosen in advance; there was no search and no local accuracy selection.

This is not a copy of the parent. Bare "But" was not scaled, because it is too common in ordinary text. "But wait" is two tokens; `wait` is already damped by the parent. PR #103 already explores a final-norm change on this same parent, so this experiment does not retune the norm.

## Edit

- Tensor: `lm_head.weight_scale` only (NVFP4 FP8-E4M3 block scales). Packed `lm_head.weight` codes are untouched.
- Factor ×0.5 on rows:
  - `ĠAlternatively` 37201
  - `Ġalternatively` 66073
  - `Alternatively` 88842
- Realized row-norm factor: 0.5 exact on all three rows (every block).
- Sparse replacement of 960 elements. Artifact SHA-256 `0ad5500a06fa7c90afe97173684977f101cf6d5a76907448623c88a618f6e99c`, 8,792 bytes.
- Parent content hash `d2574ed609c86067c8d42febe6f8869cea2615c1c2d5d820bde84e933f836b0b`.
- Seed: none. The build is deterministic.

## Reproduction

Python 3.13, torch 2.15.0+cpu, safetensors 0.8.0, numpy (needed only to write the safetensors file), deltaleague 0.5.0. CPU only. The builder reads a few kilobytes of selected rows; peak RAM is far below a full 27B load. A complete `edited.safetensors` was not written: the parent checkpoint is already 21 GiB, and a second full copy did not fit.

```sh
pip install deltaleague numpy
# CPU-only torch is enough; do not install the CUDA wheel if disk is tight.
delta competition init qwen38-27b-nvfp4-ref \
  --parent 5065b8a8-e87d-4f06-b52b-183fcb3bab8c \
  --output experiment
python research/vdimarco-27b-alternatively-head-x05/build.py \
  --parent experiment/model \
  --output experiment/submission \
  --report research/vdimarco-27b-alternatively-head-x05/results/build-report.json
delta validate experiment/submission
```

`delta patch build --parent experiment/model --modified edited.safetensors` would produce the same sparse format if a full modified checkpoint were available. The builder writes that format directly. Patch hashes should match the report.

## Verification done

- Tokenizer IDs for the three new tokens and the eight already-scaled PR #91 tokens match this checkpoint's `tokenizer.json`.
- `delta validate` passed: manifest and artifact hashes are valid. The server still has to check parent compatibility and tensors.
- All three rows have a realized NVFP4 row-norm factor of exactly 0.5.
- Edited values are finite.

## Not done / limitations

- No local inference and no public-validation estimate. Official scores will be the first measurement.
- `delta validate` does not reconstruct the parent or apply the server's allowed-prefix check. The parent already changes `lm_head.weight_scale` and is verified, so this tensor is expected to be admissible, but that is a server check.
- This does not claim that "Alternatively" is the main remaining truncation source. It is one conservative, pre-registered follow-on.

## Submission

To be added after `delta submit ./submission --title "27B: 5065b8a8 + Alternatively output rows x0.5" --pr THIS_PR_URL`.
