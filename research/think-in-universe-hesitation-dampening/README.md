# Hesitation-token damping + final-norm gain (Qwen3.5 0.8B)

Research record for one bounded weight-space experiment in the `qwen35-08b-v1` competition.
Nothing here is an official leaderboard result.

## Hypothesis

The pinned private-test generation protocol (`public-ladder500-8k-v2`) samples at
temperature 1.0, top-p 0.95, top-k 20 with presence penalty 1.5, thinking enabled, and an
8,192-token cap. Completed leaderboard rows show truncation rates of 0.44–0.83, and the
Original model's own row reports 0.746. Locally, thinking traces on public ARC-Easy items
run past 1,536 tokens as bullet-point deliberation before committing to an answer.

Two small, deterministic changes to the weights should reduce that failure mode:

1. **Hesitation-token damping.** Tied input/output embeddings for the surface forms of
   “Wait”, “Hmm”, “Actually”, “Alternatively”, “Perhaps”, “Maybe” are scaled to 0.55×.
   The hypothesis is that these tokens act as loop-entry points in the thinking trace;
   lowering their input weight and output logit should make re-entering a hesitation loop
   less likely, shortening traces without deleting reasoning steps (as forcing an early
   `</think>` does).
2. **Final-norm gain ×1.35.** With tied embeddings, scaling the final RMSNorm gain scales
   every logit by exactly 1.35 (temperature ×≈0.74). This compensates for the softened
   logits from damping and sharpens commitment to the chosen continuation. Leaderboard
   neighbours tested gain ×1.25 (val 41.45 / test 19.0) and ×1.667 (val 41.89 / test 17.2)
   against Original val 41.83 / test 18.6.

Expected effect: fewer unterminated traces and slightly more decisive final answers. The
known risk is that damping common words also removes legitimate uses (“wait for the
result”), and that ×1.35 sharpening over-commits on hard math.

## Parent and evaluation identity

| item | value |
| --- | --- |
| parent version | `0ea62f40-7cd8-4d98-9ebb-484192fa05eb` (“Original model”, no parent) |
| parent content hash | `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8` |
| parent `model.safetensors` sha256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| evaluation config | `public-ladder500-8k-v2` |
| public validation manifest | `qwen-small-public-v1`, sha256 `996573d687b0d0f1e088f1e918cd153fa0f80cff454a7e9a7ab5ac2eb374d4d2` |
| protocol seed | 20261104 (manifest) / per-example local seeds 1000+sample |
| model | Qwen3.5 0.8B, 24 layers (18 linear-attention, 6 full-attention), tied embeddings, bf16 |

## Edit and reproduction

Deterministic; no RNG is involved in producing the checkpoint.

```sh
delta competition init qwen35-08b-v1 --output experiment   # downloads the parent
cd experiment
python edit.py            # writes edited.safetensors + edit-report.json
delta patch build --parent ./model --modified ./edited.safetensors --output ./submission
delta validate ./submission
```

Parameters (`edit.py`):

| parameter | value |
| --- | --- |
| `EMBED_SCALE` | 0.55 |
| `FINAL_NORM_GAIN` | 1.35 |
| damped tokens | 16 single-token surface forms (`Wait`, `wait`, `Hmm`, `hmm`, `Actually`, `actually`, `Alternatively`, `Perhaps`, `perhaps`, `Maybe`, `maybe`; bare + leading-space where tokenized as one token). `alternatively` is two tokens and is excluded. |
| changed tensors | `model.language_model.embed_tokens.weight` (16,384 values), `model.language_model.norm.weight` (1,024 values) |
| arithmetic | float32, cast back to the parent bf16 dtype |

Expected hashes:

* edited checkpoint sha256 `164c13b1eda0aa2175bf97579d07a9d048910dd88eb401697b12dbc5838e501d`
  (1,746,942,632 bytes; parent stays at 1,746,942,600 bytes)
* sparse patch `weights.safetensors` sha256 `9607bacb09cc3a48566cffca2f4f128a421b9b94536d02f099edc806c0b1bade` (174,392 bytes)
* manifest: 2 tensors, `sparse-diff` / `replace`, parent content hash as above

`edit.py` re-reads the parent and asserts every non-target tensor is bit-identical.

## Public validation

### Harness caveat (reported as a finding)

`delta evaluate` from `deltaleague` 0.1.1 and 0.3.0 rejects this workspace with
“Public validation benchmark inventory mismatch”: the server serves a selection-only
manifest (`text`/`code` groups with `indices` + `doc_sha256`), while
`delta_cli.local.public_examples` expects `benchmark -> [example objects]`. The current
official path (`delta public-eval`) requires a Linux vLLM endpoint and the full suite
(1,110 items), which is out of scope for this machine. Two advisory harnesses were used
instead; both consume the same public documents, never the hidden test.

### A. Official-protocol likelihood subset

`mc_eval.py` reproduces the public text protocol for eight likelihood (multiple-choice)
benchmarks: native lm-eval 0.4.13 tasks, frozen dataset revisions, the first 24 official
`indices` from `validation.json`, few-shot defaults, `acc`/`acc_norm` scoring. Backend is
transformers float32 on MPS instead of vLLM bf16, so absolute values are advisory.

| benchmark | baseline acc_norm | candidate acc_norm | delta |
| --- | ---: | ---: | ---: |
| arc_challenge | 29.17 | 20.83 | −8.33 |
| arc_easy | 50.00 | 50.00 | 0.00 |
| boolq | 66.67 | 66.67 | 0.00 |
| commonsense_qa | 54.17 | 54.17 | 0.00 |
| hellaswag | 45.83 | 50.00 | +4.17 |
| openbookqa | 37.50 | 37.50 | 0.00 |
| piqa | 70.83 | 75.00 | +4.17 |
| winogrande | 54.17 | 54.17 | 0.00 |
| **equal-weight mean (8)** | **51.04** | **51.04** | **0.00** |

The subset is exactly neutral. Six benchmarks are unchanged, `hellaswag` and `piqa` move +4.17
(one document each), and `arc_challenge` moves −8.33 (two documents). With n=24 per benchmark these
moves are well inside noise (one document ≈ 4.17 points).

### B. Private-protocol generative spot check

`local_eval.py` mirrors the pinned *private-test* sampler (temperature 1.0, top-p 0.95,
top-k 20, presence penalty 1.5, thinking on, 1,536 new tokens, seeds 1000) on eight
ARC-Easy and eight GSM8K public documents. Prompts are plain and GSM8K is 0-shot locally
(the official public suite is 5-shot), so only paired differences are meaningful.

| benchmark | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| arc_easy (n=8) | 25.0 | 62.5 | +37.5 |
| gsm8k (n=8) | 25.0 | 37.5 | +12.5 |

Paired overall difference: **+25.0 points** (95% bootstrap CI −6.2 to +56.2), wins/losses/ties
6/2/8, exact sign-test p=0.289.

| metric | baseline | candidate |
| --- | ---: | ---: |
| truncation rate | 0.875 | 0.625 |
| mean output tokens | 1,434 | 1,242 |
| hesitation words in the final answer | 7.06 | 0.06 |

Directionally consistent with the hypothesis, not statistically separable at n=16. At 1,536 tokens both
arms still truncate often, so this measures commitment and trace length more than full reasoning accuracy.

## Findings

* **Hesitation damping did what it was meant to do.** In the private-protocol spot check the edited
  model used 13% fewer output tokens, cut truncation from 0.875 to 0.625, and removed hesitation words
  from its final answers (7.06 → 0.06 mentions per example). Accuracy moved from 25.0% to 50.0%
  (4/16 → 8/16), +25.0 points with a 95% CI of [−6.2, +56.2] and sign-test p=0.289: a consistent
  direction, not a separable win at n=16.
* **The public MC subset is unchanged.** On 192 official public documents scored under the public
  likelihood protocol, the equal-weight mean is 51.04 vs 51.04. This is expected: a constant logit
  scale cannot change `acc`/`acc_norm` rankings, and the damped tokens are rare in 0-shot MC prompts.
  The edit therefore looks low-risk for the ten knowledge/reasoning benchmarks, but the local evidence
  cannot claim a public-val gain.
* **Failure mode observed.** With thinking enabled at the pinned sampler, the Original model produced
  long bullet-point deliberation that often ran past the token budget without a final answer; the
  damped model committed earlier. This is the mechanism the submission tests on the private set.
* **Tooling finding.** `delta evaluate` (0.1.1 and 0.3.0) cannot run this workspace's current
  selection-only `validation.json`; `delta public-eval` is the supported path but needs a Linux vLLM
  endpoint. Both harnesses here are therefore advisory reconstructions, not the official runner.

## Limitations

* Local numbers are advisory; only the server's validation and private-test scores count.
* The likelihood subset cannot see the final-norm gain: scaling every logit by a constant
  leaves `acc`/`acc_norm` rankings unchanged. That knob is only observable in the
  generative/private protocol.
* The generative spot check truncates heavily at 1,536 tokens, so most items score 0 in
  both arms; it measures trace length and loop behaviour more than accuracy.
* No hidden test data, private generations, credentials, or weight artifacts are included.

## Submission

* Pending: submission link and verified scores will be added after `delta submit` and `delta submission results`.
