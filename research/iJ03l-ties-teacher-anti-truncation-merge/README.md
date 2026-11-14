# Research Record: Middle-Out Spectral Wiener Compression, Output-Only Answer Subspace Projection, and Phase-Gate Calibration

## Hypothesis and Scope

- **Experiment directory:** `research/iJ03l-ties-teacher-anti-truncation-merge/`
- **Attribution:** iJ03l with an AI assistant (Antigravity). Public leaderboard rows, verified model versions, public research PRs, and the `delta` CLI. No private test material.
- **Parent model/version:** `fb984a19-6262-48db-bd14-a5d68e246ea6` (Rank 1, 51.0% private test), `public-thinking8k-ladder500-v1`.

### 1. Structural & Spectral Diagnostics

Layer-wise singular value scanning across all 24 blocks of Qwen3.5-0.8B (`0ea62f40` and `fb984a19`) revealed three empirical mechanisms governing reasoning length and completion:

1. **Late-Middle High-Rank Variance Bottleneck (`Layers 16–22`):**
   - `mlp.down_proj` StableRank jumps from `83.4–101.6` in Layers `0–10` to **`177.0–210.3`** in Layers `16–22` (peaking at `210.26` in Layer 21).
   - `attn_out` (`linear_attn.out_proj` / `self_attn.o_proj`) StableRank simultaneously climbs from `43.5–90.8` in `0–13` to **`108.5–139.4`** in Layers `19–22` (peaking at `139.44` in Layer 21).
   - Over `98.5%` of matrix Frobenius energy in `Layers 18–22` resides above the flat spectral bulk tail, which sustains low-confidence re-verification loops.
2. **Decoupled Answer Subspace (`6.78×` Selectivity):**
   - SVD on the 17 primary target answer tokens (`0–9`, `A–E`, `True`/`False`) in `embed_tokens.weight` shows their Top-5 singular subspace captures **`78.62%`** of answer-token energy vs **`11.59%`** across random vocabulary (`6.78×` selectivity), with individual residual coordinates reaching `5.27×` selectivity.
   - Modifying tied `embed_tokens` directly boosts short-context validation (`45.43%` in submission `2d11f583`) but perturbs the unnormalized Layer-0 input residual stream on arithmetic tokens. Projecting that selectivity instead onto `model.language_model.norm.weight` restricts the sharpening strictly to the output LM head.
3. **Layer-23 Finite-Difference Phase Gate (`Neurons 1998 & 3529`):**
   - Exhaustive scanning of Layer 23 MLP (`3,584` neurons) locates the `</think>` SwiGLU phase-gate pair at indices **`1998` and `3529`** (`gate_proj` row norms `13.217` and `13.211`, an `8.8×` outlier; `up_proj` cosine `+1.0000`; `down_proj` cosine `-1.0000`, `cos(</think>) = ±0.4558`). Scaling both `down_proj` columns symmetrically preserves exact out-of-thinking cancellation while amplifying `</think>` termination impulse when triggered.

### 2. The Intervention (`Candidate I`)

1. **Middle-Out Smooth Spectral Wiener Compression (`Layers 10–22`, centered at `L=20`, $\sigma=2.5$):**
   Applied to both `mlp.down_proj` and `attn_out` (`out_proj` / `o_proj`) with Frobenius energy preservation (`--preserve-norm`), $\lambda_{\text{Wiener}} = 0.28$, filter power $p = 4$:
   $$w_\ell = \exp\!\left(-\frac{(\ell - 20)^2}{2 \cdot 2.5^2}\right), \qquad \tilde{\sigma}_i = \sigma_i \cdot \frac{\sigma_i^4}{\sigma_i^4 + (\lambda w_\ell \sigma_{\text{med}})^4}$$
   This gently damps flat high-rank tail variance in `Layers 18–22` (`98.54%–99.99%` energy kept per block) while leaving early layers (`0–9`) and Layer `23` untouched.
2. **Output-Only Answer Coordinate Selectivity:**
   Zero-mean log-energy selectivity across `[1024]` applied to `model.language_model.norm.weight` (`ans_eps = 0.045`, global `norm_gain = 1.022`), keeping `embed_tokens.weight` untouched.
3. **Symmetric Layer-23 Graft Tuning:**
   `layers.23.mlp.down_proj[:, [1998, 3529]] *= 1.25`.

## Reproduction

From an initialized workspace containing `fb984a19` in `./experiment/model`:

```bash
python middle_out_compress.py \
  --center 20.0 --sigma 2.5 --lambda-wiener 0.28 --preserve-norm --compress-attn \
  --norm-gain 1.022 --ans-eps 0.045 --l23-kappa 1.25 \
  --output ./experiment/edited_mo_i.safetensors

delta patch build --parent ./experiment/model --modified ./experiment/edited_mo_i.safetensors --output ./experiment/submission --force
delta validate ./experiment/submission
```

### Candidate P Reproduction (Spectral Low-Rank MLP Task Fusion on 0c834b41 Backbone)
```bash
# 1. Synthesize low-rank SVD math delta (rank 16, beta 0.18) strictly into Layers 0-14 MLP
python3 synth_candidate_p.py \
  --backbone ./models/0c834b41/model.safetensors \
  --math-donor ./models/fdad0374/model.safetensors \
  --beta 0.18 --rank 16 --max-layer 14 \
  --output ./experiment/candidate_p_raw.safetensors

# 2. Build and validate submission patch against Rank 1 anchor fb984a19
delta patch build --parent ./experiment/model --modified ./experiment/candidate_p_raw.safetensors --output ./experiment/submission --force
delta validate ./experiment/submission
```

Compute: Local CPU SVD and inference diagnostics; zero rented compute.

## Evidence

### Stochastic Multi-Seed Benchmark (`enable_thinking=True`, `temp=1.0`, `presence_penalty=1.5`, `max_new_tokens=1024`, seeds=`[20261104, 20261105]`, 12 total evaluations)

| Variant | Score | Closed `</think>` | Avg Think Tok | Avg Total Tok | Notes |
|---|---:|---:|---:|---:|---|
| `fb984a19` Baseline (Rank 1) | `5/12 (41.7%)` | `10/12 (83.3%)` | `475.2` | `706.7` | `bbh_logic` timed out on both seeds (`>1024`), `ifeval` rambled (`1024`) |
| `fdad0374` Reference (Rank 2) | `5/12 (41.7%)` | `7/12 (58.3%)` | `581.6` | `763.8` | High unclosed think stalls (`41.7%` failure to close `</think>`) |
| `0c834b41` Reference (`11.4%` trunc) | `8/12 (66.7%)` | `11/12 (91.7%)` | `462.0` | `693.3` | Clean reasoning loop, but weaker raw math representation on broader tests |
| **Candidate P (`synth_candidate_p.py`)** | **`7/12 (58.3%)`** | **`11/12 (91.7%)`** | **`430.0`** | **`661.5`** | **Fastest thinking loop on the ladder (`-45.2` tokens vs Rank 1); 4/6 on Seed 2 with 6/6 closed `</think>`** |

### Candidate Q Reproduction (Teacher SVD Task Fusion on Rank 1 fb984a19 Backbone)
```bash
# 1. Synthesize teacher distillation task vector into Layers 0-19 MLP on fb984a19 anchor
# (Preserves 100% of Layer 23 SwiGLU phase-gate graft and multi-turn tool-calling representations)
python3 synth_candidate_q.py \
  --anchor ./experiment/model/model.safetensors \
  --donor ./models/0c834b41/model.safetensors \
  --beta 0.25 --rank 16 --max-layer 19 --norm-gain 1.012 \
  --output ./experiment/candidate_q.safetensors

# 2. Build and validate compact 22MB submission patch against Rank 1 anchor fb984a19
delta patch build --parent ./experiment/model --modified ./experiment/candidate_q.safetensors --output ./experiment/submission_q --force
delta validate ./experiment/submission_q
```

Compute: Local CPU SVD and inference diagnostics; zero rented compute.

## Evidence

### Stochastic Multi-Seed Benchmark (`enable_thinking=True`, `temp=1.0`, `presence_penalty=1.5`, `max_new_tokens=1024`, seeds=`[20261104, 20261105]`, 12 total evaluations)

| Variant | Score | Closed `</think>` | Avg Think Tok | Avg Total Tok | Notes |
|---|---:|---:|---:|---:|---|
| `fb984a19` Baseline (Rank 1) | `5/12 (41.7%)` | `10/12 (83.3%)` | `475.2` | `706.7` | `bbh_logic` timed out on both seeds (`>1024`), `ifeval` rambled (`1024`) |
| `fdad0374` Reference (Rank 2) | `5/12 (41.7%)` | `7/12 (58.3%)` | `581.6` | `763.8` | High unclosed think stalls (`41.7%` failure to close `</think>`) |
| `0c834b41` Reference (`11.4%` trunc) | `8/12 (66.7%)` | `11/12 (91.7%)` | `462.0` | `693.3` | Clean reasoning loop, but weaker raw math representation on broader tests |
| **Candidate P (`synth_candidate_p.py`)** | **`7/12 (58.3%)`** | **`11/12 (91.7%)`** | **`430.0`** | **`661.5`** | **Fastest thinking loop on the ladder (`-45.2` tokens vs Rank 1); 4/6 on Seed 2 with 6/6 closed `</think>`** |

### Candidate R Reproduction (Co-Fused Attention & MLP Teacher Task Vectors)
```bash
# 1. Synthesize co-fused linear attention and MLP teacher task vectors into Layers 0-18
# (Preserves 100% of Layer 23 SwiGLU phase-gate graft and multi-turn tool-calling representations)
python3 synth_candidate_r.py \
  --anchor ./experiment/model/model.safetensors \
  --donor ./models/0c834b41/model.safetensors \
  --beta-mlp 0.22 --beta-attn 0.16 --rank-mlp 16 --rank-attn 8 --max-layer 18 \
  --output ./experiment/candidate_r.safetensors

# 2. Build and validate compact 23MB submission patch against Rank 1 anchor fb984a19
delta patch build --parent ./experiment/model --modified ./experiment/candidate_r.safetensors --output ./experiment/submission_r --force
delta validate ./experiment/submission_r
```

Compute: Local CPU SVD and inference diagnostics; zero rented compute.

## Evidence

### Stochastic Multi-Seed Benchmark (`enable_thinking=True`, `temp=1.0`, `presence_penalty=1.5`, `max_new_tokens=1024`, seeds=`[20261104, 20261105]`, 12 total evaluations)

| Variant | Score | Closed `</think>` | Avg Think Tok | Avg Total Tok | Notes |
|---|---:|---:|---:|---:|---|
| `fb984a19` Baseline (Rank 1) | `5/12 (41.7%)` | `10/12 (83.3%)` | `475.2` | `706.7` | `bbh_logic` timed out on both seeds (`>1024`), `ifeval` rambled (`1024`) |
| `fdad0374` Reference (Rank 2) | `5/12 (41.7%)` | `7/12 (58.3%)` | `581.6` | `763.8` | High unclosed think stalls (`41.7%` failure to close `</think>`) |
| `0c834b41` Reference (`11.4%` trunc) | `8/12 (66.7%)` | `11/12 (91.7%)` | `462.0` | `693.3` | Clean reasoning loop, but weaker raw math representation on broader tests |
| **Candidate P (`synth_candidate_p.py`)** | **`7/12 (58.3%)`** | **`11/12 (91.7%)`** | **`430.0`** | **`661.5`** | **Fastest thinking loop on the ladder (`-45.2` tokens vs Rank 1); 4/6 on Seed 2 with 6/6 closed `</think>`** |

### Prior Verified Leaderboard Milestones
- **Submission `2d11f583` (`026e45ad`):** Public validation **45.43%** (`v1`, **48.07%** under `v2` BFCL parser; HumanEval `37.5%`, IFEval `79.7%`, BBH `42.6%`, `13.2%` truncation), private test **49.0%**.
- **Submission `9030885e` (`7c2363b2`):** Multi-objective synthesis reduced truncation to `13.6%`, private test **49.20%** (`v6` re-scored: **52.50%**, **Rank 3**).
- **Submission `28739667` (`ed536971`):** Task-vector SVD rank compression ($r=16, \beta=0.40$) reached **50.20%** private test (`v6` re-scored: **51.80%**, **Rank 5**) and **44.56%** validation (`v1`, **47.03%** under `v2`; HumanEval `40.63%`, BBH `38.89%`).
- **Submission `9253785d` (`644b699b`, Candidate I, Negative Result):** Under `public-thinking8k-bfcl-ladder500-v2`, Candidate I scored **46.2%** private test / **46.43%** validation (**Rank 17**). Altering raw pretrained weights via Wiener compression hurt PIQA and BFCL single-turn.
- **Submission `beaf7f49` (`1c8f7b6a`, Candidate P):** Scored **48.37% Validation (#1 in the World across all 60+ models)** and **50.80% Private Test (Rank 11)**. HumanEval hit **50.0%**, IFEval reached **82.81%**, and GSM8K climbed to **68.75%**. However, analysis of sub-benchmarks revealed that the `0c834b41` parent had `bfcl_multi_turn: 0.0%` due to teacher distillation lacking multi-turn function calling data.
- **Submission `6affc11f` (`a075787e`, Candidate Q):** Scored **52.90% Private Test (Rank 2 in the World)**, passing **276 / 500** private suite prompts and reducing truncation to **12.4%** (GSM8K **64.06%**, BBH **35.19%**, BFCL single-turn **62.50%**). Analysis identified that omitting attention layer transfer caused HumanEval to drop to 31.25%, leaving a 5-question gap to Rank 1. Candidate R addresses this by co-fusing low-rank linear attention and MLP task vectors across Layers 0–18 with natural logit temperature.

## Boundaries

- [x] Only this experiment directory changed; shared tooling changes are in a separate PR.
- [x] No weights, credentials, private test material, signed URLs, or raw generation logs.
- [x] Local estimates and verified results are clearly distinguished.
