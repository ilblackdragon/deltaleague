# Warm-start LoRA with an isolated final-norm gain

**Hypothesis.** A short continuation from our round-one adapter improves math. An isolated final RMSNorm gain can recover instruction following while preserving most of that math improvement.

Aaron approved the experiment and competition submission. Codex ran the experiment using our round-one adapter, training-split data, original instructions and logic/tool prompts, and the pinned public evaluation. The candidate derives from the original Qwen3.5-0.8B parent through our round-one adapter. The round-two merge and the full three-part stopping edits served as comparisons.

## Identity and recipe

- Model `qwen35-08b-v1` and original parent `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`.
- Parent content hash `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Round-one adapter documented in [PR #109](https://github.com/ilblackdragon/deltaleague/pull/109), submission `b94b64fe-1aac-4394-b7fa-6a7be1d3f8b0`.
- Evaluation `private1000-v6-thinking8k-20261111`. The pinned public suite has 1,110 items across 18 benchmarks.
- Select 320 examples from our Phase B pool. Train one seed, `20261127`, for 20 steps at batch 16 with completion-only loss, cosine schedule, peak learning-rate setting 2.5e-5, rank 16, alpha 32 and max length 8,192. Initialize all 150 targets from the verified round-one factors.
- Apply `w' = 1.4(1+w)-1` to the final zero-centered RMSNorm weight. The rest of the warm-start checkpoint, including the embedding table, is bitwise identical. Credit for the stopping-edit comparison goes to [PR #91](https://github.com/ilblackdragon/deltaleague/pull/91).

## Reproduction

See [TRACE.md](TRACE.md) for the frozen inputs, serving setup and validation steps. Use the original parent directory `model-orig` and the verified round-one submission directory `runs/lora1/submission`. Run the commands from the experiment root after placing the included Phase B and warm-start tools there in the layout described in the trace.

```bash
python r2/pilot/pilot.py prepare runs/warm-pilot-20261127
python r2/pilot/train.py --parent model-orig --data runs/warm-pilot-20261127/train.jsonl \
  --out runs/warm-pilot-20261127/train --submission runs/warm-pilot-20261127/adapters \
  --initial-adapter runs/lora1/submission --checkpoints 10 20 --seed 20261127 \
  --lr 2.5e-5 --rank 16 --alpha 32 --max-len 8192
python reconstruct_eval.py model-orig runs/warm-pilot-20261127/adapters/step-20 \
  runs/warm-pilot-20261127/model-step-20
python final_norm.py runs/warm-pilot-20261127/model-step-20 runs/warm-step-20-norm14
python build_patch.py model-orig runs/warm-step-20-norm14/model.safetensors submission-dense
delta validate submission-dense
```

The trainer uses ROCm and the Qwen linear-attention kernels. Reconstruction, patch building and public scoring use the isolated evaluator environment. Use that environment for packaging and supply the matching public validation manifest and evaluation config when running the suite.

## Evidence

| Model | Local public mean |
| --- | ---: |
| Original parent | 40.08 |
| Round-one LoRA | 42.25 |
| Round-one LoRA + full Phase A stopping edit, norm 1.4 | 46.06 |
| Warm-start step 20 + norm 1.4 only | **46.97** |

The candidate gained **4.72** points over round one, with a paired exploratory 95% interval of **+2.4 to +7.0**. Its largest regression was MMLU-Pro at **−4.29**. HumanEval gained **25.00**, BBH **18.52**, and Math500 **14.06**. GSM8K fell **1.56** and IFEval gained **4.69**. Against the Phase A stack the change was **+0.91**, with an interval of **−1.4 to +3.2**. The interval resamples items within benchmarks and excludes run-to-run generation variability and candidate selection effects.

On the same 96-item development set the candidate scored 47/96 against 39/96 for unedited warm-start step 20. That gain came from harder math and instruction following. GSM8K fell from 17/32 to 15/32. Boosting the end-of-thinking embedding alone and using the full three-part stopping edit reduced truncation, while instruction passes were flat or lower. This motivated carrying forward the norm-only candidate.

## Submission

The exact dense replacement patch modifies 151 tensors. Delta reconstruction was checked bit for bit against the evaluated checkpoint. Array values and signed upload URLs are kept out of this PR.

[Delta League submission](https://deltaleague.ai/submissions/b2d0dc5f-b1c0-4538-8d5a-689f8bf326cc/) verified on November 28, 2026. The official private test score was **29.2** (standard error **1.44**) and official public validation was **46.14**. The entry ranked **83rd of 99** at the time of the check. Round one scored **20.4** private and **41.99** validation. The original scored **17.4** private and **40.29** validation.
