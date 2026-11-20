# Port #77's seed-soup delta onto eff1e1b7: eff1e1b7 + (soup2 − fb984a19)

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `eff1e1b7-18ef-4c77-83ef-25fdcc027e75` (Congoss #80, private 54.3).

## Hypothesis

Two edits on `fb984a19` touch almost disjoint parts of the model:

- **#80 (`eff1e1b7`, 54.3):** a 25% rollback of the layer-23 phase-gated neurons 1998 and 3529. It changes 2,043–2,047 entries in each of three layer-23 MLP tensors and nothing else.
- **#77 soup2 (`5d4b0b84`, 53.3, truncation 12.6%, lowest among the top entries):** the uniform average of two 8-step length-shaped GRPO runs. It changes 255 tensors with small updates across the language model.

If the two effects are roughly independent, their sum should keep both. This entry adds the soup2 task vector, unchanged, onto `eff1e1b7`. It complements SpacexCeo's `a6eb9f13` (54.3), which re-trained two new seeds on `eff1e1b7` rather than reusing #77's weights.

## Method

- **Build:** `port_delta.py fb984a19.safetensors soup2.safetensors eff1e1b7.safetensors OUT.safetensors`. It computes `OUT = eff1e1b7 + (soup2 − fb984a19)` in float64 and stores the parent dtypes.
  - Outside the three layer-23 MLP tensors, OUT is bit-identical to soup2.
  - Inside them, both changes are added. soup2 had moved 725, 7 and 75 of #80's edited entries in `down_proj`, `gate_proj` and `up_proj` respectively.
- **Check:** no training and no GPU. `delta patch build` against `eff1e1b7` changes 255 tensors, all under `model.language_model`. `delta validate` passes.

| Artifact | sha256 |
|---|---|
| ported weights | `c86821eb8d6cca66cb025443419169478b975d3e3a2e770b35781eaa8027f44b` |
| patch weights | `568ae9137bb627d7fb13bd20ea21f5e25342b8e36b4adbb72680b3612443db87` |
| patch manifest | `74ecbbadc251b2b15fc36ee6979c0ad44a11d1a3461c1fd8331963d8f42d5f68` |

## Local evidence

There is none beyond the build checks above.

- Both ingredients are verified entries, and the sum differs from each of them only where described.
- In #69, #74, #77 and #78 our local probes did not predict the private ordering, so the private test is the measurement.
- With a test SE of about 1.6, a result within ±1.6 of 54.3 is the expected null outcome.

## Credit

Credit goes to:

- Congoss: `eff1e1b7` (#80);
- SpacexCeo: `fb984a19`;
- roshaans: `grpo.py` and the recipe (#58);
- dawgster: the layer-23 graft (#66).

## Official result

Added after `delta submit … --pr`.
