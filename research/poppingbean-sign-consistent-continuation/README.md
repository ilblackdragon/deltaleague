# fdad0374 + sign-consistent part of roshaans' continuation delta

Competition: `qwen35-08b-v1` (Qwen3.5-0.8B). Parent: `fdad0374-29f3-49c6-911d-d0ce9531f7a6` (dawgster, #66: `ac147fa6` + phase-gated ready graft, private test 50.8).

## Hypothesis

`fb984a19` (SpacexCeo, private test 51.0) adds roshaans' continuation delta Δr = `0c834b41` − `d5fc95ea` (9B-teacher distillation + 30 GRPO steps) to `fdad0374`. The final norm and the layer-23 graft are kept. A check in `c1_merge.py` confirms that recipe: a plain `fdad0374` + Δr matches `fb984a19` on 741,349,919 of 741,349,920 elements in the tensors it changes.

`fdad0374` already contains our merged GRPO delta D = `ac147fa6` − `9492e219` (#62). In #62's analysis, two GRPO runs from the same parent moved the same weights far more often than chance, but agreed in sign only 52% of the time. Most single-ulp bf16 moves are optimiser noise. If Δr is added where it points against D, it mostly re-randomises weights that D already moved.

**Edit:** for each element, add Δr only where D is zero or has the same sign as Δr. Drop Δr where the two signs disagree.

| | elements |
|---|---|
| Δr kept (D zero or same sign) | 42,167,274 |
| Δr dropped (opposite sign) | 26,229,778 (38%) |
| unchanged from `fdad0374` | final norm and the 3 layer-23 graft tensors |

Risk: the dropped part may carry real signal from the distillation. `0d7af43b`, a three-source TIES merge with sign election, scored 46.0, so sign-based filtering is not reliably helpful.

## Reproduce

```
python c1_merge.py BASE_9492e219 AC147FA6 FDAD0374 D5FC95EA 0C834B41 FB984A19 OUT.safetensors
delta patch build --parent ./model --modified ./OUT.safetensors --output ./submission   # parent fdad0374
delta validate ./submission
```

The script works in float64 in 8M-element chunks, so peak memory stays small. Results are stored in each tensor's original dtype.

| Artifact | sha256 |
|---|---|
| merged weights | `06c672ee61aed042465b1b5f7868bfb070a61c6f0a08d96588903ef50237e2ff` |
| patch weights (253 tensors vs `fdad0374`) | `049231b9e4ac5d1070d5dfc5d4b11a80657c390e19c4d99080e23477813e7dd7` |
| patch manifest | `2940c0a869b1a743a9d32ec54e6bfe597679962f8aa227a93a50e209876a12e3` |

## Evidence

There is no local accuracy estimate. The only local machine is an AMD GPU without vLLM, which is too slow to separate candidates 1–2 points apart. The entry is submitted as a direct test of the sign-consistency hypothesis against `fb984a19`.

## Credit

Credit goes to:

- roshaans: `d5fc95ea` and `0c834b41`;
- SpacexCeo: the idea of transplanting that delta onto `fdad0374` (`fb984a19`), and `18db9b31`;
- dawgster: `fdad0374`;
- ma4ypic4y: `9492e219`.

The ×1.05 final-norm step comes from #43 and #55.
