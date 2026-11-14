# Extrapolating the seed-soup direction: fb984a19 + 1.3 × (soup2 − fb984a19)

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `fb984a19-6262-48db-bd14-a5d68e246ea6` (SpacexCeo, private test 53.4).

## Hypothesis

#77's soup2 (`5d4b0b84`) is the uniform average of two independent 8-step runs of length-shaped GRPO on `fb984a19`. It scored 53.3 privately, against 52.2 for its single-seed ingredient c8 (`4b8a20a6`, #74). Truncation fell from 14.2% (parent) to 13.2% (c8) and then to 12.6% (soup2).

Averaging two seeds should leave the direction both runs agree on and shrink the seed-specific noise. Because souping already removed some noise, the soup's task vector Δ = soup2 − fb984a19 can be scaled up more safely than a single run's. We test a mild step: **fb984a19 + 1.3 · Δ**.

## Method

- **Build:** `extrapolate.py fb984a19/model.safetensors soup2.safetensors 1.3 OUT.safetensors`. It computes `P + 1.3·(M − P)` in float64 and stores each tensor in its original dtype. Tensors equal in both files are copied unchanged. The result changes 255 tensors, the same set as soup2. No training and no GPU.
- **Check:** `delta patch build` against `fb984a19` changes 255 tensors, all under `model.language_model`. `delta validate` passes.

| Artifact | sha256 |
|---|---|
| extrapolated weights | `90c6d56e2aa22d305edef1684a9434dcc40bdbda7409ebaa8cbf77677ebf8fd2` |
| patch weights | `242fc994229dc6996cc720c6b3b070bda89a60fc9ba8e61e545eab83d61170df` |
| patch manifest | `0ed3bfe98f6a8f2e5d75057f59b0603836e7adb953a7c54e4819a2c00fe08973` |

## Local evidence (breakage check only)

`nll_check.py` computes the mean token NLL of 16 fixed correct answers (11,756 tokens), taken from c8's probe answers, on CPU:

| | `fb984a19` | soup2 | **1.3 × soup2** |
|---|---|---|---|
| mean NLL | 0.5567 | 0.5565 | 0.5565 |

The edit is small: the median relative change per tensor is about 2e-4. This check only shows the model is not broken. It says nothing about accuracy.

No probe or public-suite run was made: no GPU was rented for this entry. In #69, #74 and #77, local probe differences did not predict the private ordering. The private test is the measurement here.

## Related observation: `a075787e` (#68, Candidate Q)

We considered averaging soup2 with iJ03l's `a075787e` (52.9). Measured against `fb984a19`:

- its MLP change alters 0.67% of MLP elements, each by about one bf16 step;
- its main change is the final norm × ~1.012 (range 1.0097–1.0157);
- the two task vectors are nearly orthogonal (cosine ≈ 0 on the 60 shared tensors).

A cross-merge would therefore amount to "soup2 + final norm × 1.012", so we did not pursue it.

## Credit

Credit goes to:

- roshaans: `grpo.py`, the recipe and the probe (#58);
- SpacexCeo: `fb984a19`;
- task-vector arithmetic (Ilharco et al., 2023).

## Official result

Submission `f8a6b4ab-186a-477d-8335-94290757cc12`, version `266b4d90-7d42-486a-a894-af9b3c76cefd`, verified.

| | `fb984a19` (parent) | soup2 (#77) | **This entry: 1.3 × soup2** |
|---|---|---|---|
| Private test | 53.4 | 53.3 | **53.1** (rank 3; SE 1.58) |
| Public validation | 47.40 | 47.08 | 46.61 |
| Test truncation rate | 14.2% | 12.6% | 13.8% |

**Reading:** a negative result, within noise. Scaling the soup's direction by 1.3 lowered the private score by 0.2 and pushed truncation back up by 1.2 points. Most of the 52.2 → 53.3 gain from c8 to soup2 does not look like a direction that keeps paying off when amplified. It is more consistent with averaging away noise, plus seed luck. soup2 (λ = 1.0) remains our best entry.
