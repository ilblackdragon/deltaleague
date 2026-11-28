# Reproduction trace

## Layout and data

Copy the included Phase B scripts to `r2/`, the `warm-pilot` scripts to `r2/pilot/`, and the other Python files to the experiment root before running the README commands. Put the trainer’s root, `r2`, `pylib-fla` and `pylib` on its `PYTHONPATH`. Use the clean public evaluator environment without trainer libraries.

Recreate the training-only artifacts from round one using [PR #109](https://github.com/ilblackdragon/deltaleague/pull/109). Derive the Phase B pool with the bundled preprocessing and generation scripts. Split off the 128 math holdouts with `prep_data.py`, then use `r2_data.py math` to select 700 GSM8K and 700 math items. Create the 800 hint prompts with `hint_data.py prep data/r2/math-hints.jsonl --heldout data/heldout.jsonl --n 800`. Make MC train and holdout inputs with `r2_mc.py` and rehearsal prompts with `r2_rehearsal.py`. The generation runner reads those files as `data/r2/{math,math-hints,mc,mc-heldout,rehearsal}.jsonl`.

Run `r2/generate.sh runs/r2-round2-v3/generation FAMILY` for math, mc, rehearsal, and hints on the round-one LoRA, with one worker owning the GPU at a time. Build the mix with `r2_data.py mix` from the three selected files and `math-repair.jsonl`, excluding `data/heldout.jsonl` and `data/r2/mc-heldout.jsonl`. The warm-start prep verifies the resulting mix against its recorded hash before sampling. Preserve the selected inputs from the reported run for bitwise downstream reproduction. Generation may vary across serving runs.

The 1,600-row mix contains 560 math, 480 multiple-choice, 400 instruction/coding, and 160 logic/tool examples. The 320-row sample balances GSM8K and harder math at 272,208 and 272,162 completion tokens respectively. The 96-item development set has 32 items each from GSM8K, harder math and original instructions. It is disjoint by ID and normalized question from the pilot training sources and existing holdout.

## Configuration and hashes

- Pool SHA-256 `8488c9c04de15e019e88fee6ef5eadc8315a109d9529373097c265b0e2521a39`
- Pilot training JSONL SHA-256 `00b6be98fde6d3a66e0e6d9e687ab03c06fe32eed98392bcece775bd00ef2af0`
- Development JSONL SHA-256 `fe0808f1ac9fc11743f6caaf839fb85cd5a2c030431833626a5aa31f3b4eeeb3`
- Evaluated model file SHA-256 `d6879314e1f4f0a6b2d95618b88c6148f27fbe753afe1f4c8abb96b46bc03d1b`
- Dense patch manifest SHA-256 before adding the PR association `8c7c312d314279cc8f1d7b11285c4d90c255a9dd099c6d22b3390dd0141ddf8e`
- Dense patch weights SHA-256 `c50f2a2abf0814a638d797e9ba5d9a3f120f7a69eb10b02307274eb69cb3f27c` (994,065,224 bytes)

Rollout settings use thinking on, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5 and an 8,192-token cap. The paired development probes use `r2_data.py probe --samples 1 --paired --conc 64` and the same frozen item order for every checkpoint. The full public evaluation uses the pinned settings and BFCL adapter, a 34,816 context and 32 server sequences. It ran on AMD ROCm with the pinned image in `serve.py`. Training of the warm-start candidate took 31 minutes. The candidate’s public suite took about 39 minutes after model startup.
