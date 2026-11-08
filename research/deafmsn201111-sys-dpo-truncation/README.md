# Leader + self-distilled SFT LoRA + anti-truncation DPO LoRA

## Hypothesis and access

The private test runs with thinking on and an 8192-token cap. The leader `cd0b95bb` truncates about 16% of private-test answers. On public MATH train items it truncates 35%, and almost half of those truncations happen *after* `</think>`, in loops during the answer phase. The hypothesis was that preference training could teach the model to finish (chosen = a correct, finished trace; rejected = a truncated trace of the same prompt) and turn some truncations into scored answers.

Work was done by a human with an AI coding agent (Claude Code). We used only public resources: the leaderboard API, public checkpoints, and the **train** splits of public datasets for all training data. We did not access private test items or validation answers.

## Parent and configuration

- Parent: `cd0b95bb-c5c0-4656-b9be-ad6482f04075` (content hash `30f14773…52f1`); model `qwen35-08b-v1`; evaluation config `public-ladder500-8k-128-v1`.
- Final weights: `leader → + LoRA v1 (SFT) → cast bf16 → + LoRA DPO-b → cast bf16`. Each LoRA step is `W.float() + (alpha/r)·B@A`, the same rule as Delta's LoRA reconstruction.
- Submitted as `dense-diff` / `replace`: the 150 changed linear weights, with the other 338 tensors untouched.

## Pipeline (all compute: Kaggle T4 ×2, vLLM 0.33 fp16, sampler identical to the private test: T=1, top_p 0.95, top_k 20, presence_penalty 1.5, max_tokens 8192, thinking on)

1. **Proxy** (`code/kaggle_calibrate.py`): 500 public eval items, 100 each from GSM8K test, MATH-500, MMLU-Pro test, ARC-Challenge test and CSQA validation (seed 1234). Five leaderboard checkpoints gave Pearson 0.95 vs the official scores. The proxy is easier than the private test: the leader scores 54.0 here vs 37.2 officially. The scorer is format-tolerant for MC (`code/rescore_lib.py`).
2. **SFT LoRA v1** (`code/kaggle_gen.py`, `select_traces.py`, `lora_train.py`, `kaggle_train.py`): 2 samples per item from the leader on 2300 train items (GSM8K, MATH-lighteval train, ARC-C train, CSQA train, MMLU auxiliary_train; seed 777). We kept 1316 traces that are correct, finished and loop-free, with a short `\boxed{}` answer and at most 6000 tokens, choosing the median-length trace per item. Training: r16, alpha 32, 150 modules (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`), lr 1e-4, 1 epoch, 82 steps, completion-only loss.
3. **Pairs** (`code/kaggle_gen2.py`, `make_pairs.py`): 4 samples per item from leader+LoRA v1 on 1500 MATH-lighteval train items (levels 2–5) and 300 GSM8K train items (seed 2026). This gave 248 pairs from 173 items. Chosen is correct, finished, boxed and ≤ 6000 tokens. Rejected is truncated (answer-phase truncations first), cut to think+1024 tokens.
4. **DPO-b** (`code/dpo_train.py`, `kaggle_dpo.py`): DPO + 0.2·NLL on chosen (RPO-style). β 0.1, lr 5e-5, r16, alpha 32, 2 epochs, 62 steps, grad-accum 8. The reference log-probs are precomputed from leader+LoRA v1. A sister run DPO-a (lr 2e-5) is also reported below.
5. **Build** (`code/build_dense.py`): merges both LoRAs onto the pulled parent, writes the dense-diff, and checks it with `delta_cli.artifacts.reconstruct`. Result: 0 mismatches, and the other 338 tensors equal the parent.

Expected hashes:

| file | sha256 |
|---|---|
| LoRA v1 `lora.safetensors` | `8a8efa44b4cd164796a3c52d952b443e9e21047491ac92ffed76cd5d148382ee` |
| LoRA DPO-b `lora.safetensors` | `a6990e00854570a38575ac4dbf395471fbd5139db22eae21fb9bfd039e1843b4` |
| submission `weights.safetensors` (994 MB) | `808f9185d1cc6645ae72611aa507b7b4154db0fb2761f9e1af99b775673108a9` |

## Evidence (local estimates on the 500-item proxy, NOT official results)

Each row has n = 500. CIs are a paired bootstrap 95%. Full output is in `proxy_compare.txt`.

| model | acc | Δ acc vs leader | truncated | Δ trunc vs leader |
|---|---|---|---|---|
| leader `cd0b95bb` | 54.0 | — | 10.4% | — |
| + LoRA v1 (SFT) | 55.2 | +1.2 [−3.0, +5.6] | 8.2% | −2.2 [−4.2, −0.2] |
| + LoRA v1 + DPO-a | 53.2 | −0.8 [−5.0, +3.8] | 4.2% | −6.2 [−8.8, −3.8] |
| **+ LoRA v1 + DPO-b (submitted)** | **54.2** | **+0.2 [−4.4, +4.8]** | **5.4%** | **−5.0 [−7.4, −2.6]** |

## Negative findings and limitations

- DPO roughly halves truncation, and the drop is significant, but accuracy does not move. Of the 52 proxy items the leader truncates, DPO-b finishes 31, but only 12 of those are correct. Truncations sit mostly on problems the 0.8B model cannot solve, so finishing them mostly produces wrong answers.
- The submission bets that the harder private test, where truncation is higher, rewards the truncation cut more than the proxy does. We expect at most about +1–3, which is within the test's noise (SE ≈ 2.2).
- A disproved earlier hypothesis: "the leader under-thinks on multiple-choice". With format-tolerant scoring on 1000 train MC items, a longer-thinking checkpoint was not better. The apparent gap came from the leader skipping `\boxed{}` on 7% of MC answers, plus 100-item noise.
- Kaggle generations are deterministic for fixed weights and seed, so re-running a baseline adds no information.
- Generated traces, preference pairs and weights are not included (repository policy). The scripts regenerate them.

## Submission

Submission ID: _to be added after `delta submit`._
