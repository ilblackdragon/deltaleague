# Delta League sprint — shared brief for all agents (read fully)

## Objective
Beat **28.20% private test** on deltaleague.ai, competition `qwen35-08b-v1` (Qwen3.5-0.8B, thinking model), while public
validation stays >= 39.5%. Rank is decided ONLY by the private test score (500 items, server SE ~2.0 points).
Only the orchestrator submits to Delta League. Agents NEVER run `delta submit`, `delta login`, or any account action.

## Server evaluation config (evaluation.json, public-ladder500-8k-v2)
vLLM, thinking ON, temperature 1.0, top_k 20, top_p 0.95, presence_penalty 1.5 (on generated tokens only, applied to logits
before temperature/top-k/top-p), max_tokens 8192. Answer = text after the last `</think>`; no `</think>` => score 0.

## Leaderboard facts (private test, truncation_rate = share of generations hitting 8192 tokens)
| version | edit | private | trunc | out tokens |
|---|---|---|---|---|
| 0ea62f40 | original model | 18.6 | 0.746 | 3.58M |
| ab592312 | **leader**: 11 hedge rows e<-0.6e+0.4*mean(E), final norm (1+w)*1.4 | **28.2** | 0.526 | 2.95M |
| 9ae2205f | leader + published repetition-direction edit | 26.2 | 0.504 | 2.84M |
| 86968401 | leader rows (11 @0.6 mean) with TOTAL gain 1.25 (not 1.75; verified by diff) | 26.0 | 0.548 | 3.03M |
| 8dc7d17c | same 7 rows as new leader, pure scale 0.4, g=1 | 24.6 | 0.572 | 3.19M |
| 67c994de | 3 rows (' Wait','Wait','Hmm') e<-0.4e+0.6mu, g=1 | 22.4 | 0.598 | 3.20M |
| a155dc2f | published sparse MLP edit (layers 15,19 down_proj) | 19.0 | 0.716 | 3.52M |
| 4fe2f3d6 | final norm gain 1.25 | 19.0 | 0.756 | 3.57M |
| 0ffb5aef | `</think>` row gain x1.5 | 17.6 | 0.44 | 2.06M |
| d17ce35f | final norm gain 1.667 | 17.2 | 0.826 | 3.65M |
| e75e2ed5 | published edit x1.5 | 13.8 | 0.722 | 3.55M |
Pending on the board (others): leader with 7 Wait/Hmm rows at 0.5x; "dampen + norm 1.4 + think close 1.3".
Take-aways: truncation is the dominant failure mode, but forcing `</think>` early (0ffb5aef) kills accuracy; gain alone
1.25 does nothing, 1.667 *increases* truncation (low temperature + presence penalty -> loops?). Dampen+gain is synergistic.

## Leader edit, exactly (verified by diffing weights vs baseline)
Model is tied (`tie_word_embeddings: true`): `model.language_model.embed_tokens.weight` is also the LM head.
Final norm is `x/rms(x) * (1 + w)` (`model.language_model.norm.weight`), so multiplying (1+w) by g scales ALL logits by g.
Rows changed (token ids): 3655 'Ġwait', 10380 'ĠMaybe', 13428 'ĠWait', 13784 'Wait', 20734 'Maybe', 32645 'ĠActually',
37201 'ĠAlternatively', 50821 'Actually', 77264 'Hmm', 85152 'ĠHmm', 88842 'Alternatively'; each e <- 0.6 e + 0.4 mu,
mu = mean of all embedding rows. Logit of such a token: g*(0.6 h.e_t + 0.4 h.mu).

## Infrastructure
Local sandbox: /home/user/delta (4 CPU, 15 GB RAM, no GPU). venv: `source /home/user/delta/.venv/bin/activate`.
Weights locally: /home/user/delta/exp-base/model (original), /home/user/delta/exp-leader/model (leader). Do not modify them.
Rented GPU box (RTX 5090 32GB, 32 CPU cores, 60 GB RAM), reachable ONLY through `python /home/user/delta/tools/gpu.py`:
- `gpu.py sh "<bash>" [timeout]` run a short command (returns output). `gpu.py bg "<bash>"` start detached.
- `gpu.py put LOCAL dl/REMOTE`, `gpu.py get dl/REMOTE LOCAL` (paths relative to /home/clore).
- Never print or copy the credential file. Never kill `runner.py` or other agents' processes. Never touch `~/dl/versions`.
- GPU box python env: `$HOME/venv/bin/python` (vllm 0.30, torch, transformers 5.22, datasets, math_verify).
- The GPU itself is owned by the serial queue runner (vLLM uses ~29 GB). CPU analysis on the box is fine
  (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6`), keep it to <= 6 threads and <= 12 GB RAM per agent.
GPU queue (`python /home/user/delta/tools/q.py`): `submit NAME SPEC.json`, `status`, `result NAME [DIR] [--texts]`, `log NAME`.
Each job builds a candidate with `~/dl/apply_edit.py` from the ORIGINAL baseline and evaluates it with `~/dl/eval_vllm.py`
on the proxy set (~8-10 min per job). Spec fields (JSON):
```
{"tokens": {" Wait": 0.6, "Wait": 0.6, ...},   # surface string -> s ; leading space = 'Ġ'; must be a single vocab token
 "mode": "mean",                                # 'mean': e<-s e+(1-s) mu (leader) | 'scale': e<-s e
 "gain": 1.4,                                   # final norm (1+w) multiplier
 "bias": {"</think>": 0.0},                     # e_t <- e_t + b*mu/|mu|^2  (adds ~b*(h.mu)/|mu|^2 to that logit)
 "rep_alpha": 0, "rep_dir": "upstream",         # published sparse edit scaled by alpha (files in ~/dl/upstream if present)
 "samples": 1, "max_tokens": 8192, "prompts": "proxy400.json"}
```
or `{"model": "versions/<id>/model"}` to evaluate an existing checkpoint. A spec reproducing the leader:
tokens = the 11 strings above at 0.6, mode mean, gain 1.4. If you need an edit type apply_edit.py does not support,
write your own builder as `~/dl/agents/<tag>/build.py` that writes a full model dir, then submit `{"model": "agents/<tag>/cand_x"}`
(copy tokenizer/config files from versions/0ea62f40-7cd8-4d98-9ebb-484192fa05eb/model). Keep total keys/shapes/dtypes identical.
Proxy set `~/dl/proxy400.json`: 320 public prompts (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-Challenge 40), 1 sample each,
fixed seed per prompt. Result JSON: `summary` (per bench + ALL: acc, trunc, tokens, hedges, acc_closed) and `rows` per prompt
(score, closed, tokens, truncated, hedges); `--texts` also fetches full generations (`*.texts.jsonl`).
Calibration jobs `c_leader, c_base, c_g175, c_think15, c_damp04, c_g1667, c_g125, c_rep` evaluate leaderboard checkpoints
with known private scores (table above); use them as ground truth for the proxy. They finish over the next ~80 min.
**GPU budget: at most 3 queued jobs per agent at any time, 4 in total for this round**; name jobs `<tag>_<short>`.
Prefer cheap analysis (weights, logits on fixed traces, teacher forcing on CPU) to burning queue slots.

## Rules
Data truth: never invent numbers; label local proxy numbers as advisory. No private test data. No weights in git.
Write your work to /home/user/delta/agents/<tag>/ (code + REPORT.md). REPORT.md must end with a short
"Recommendation" section: concrete spec JSON(s) you believe beat the leader on PRIVATE, with the math/evidence and risk.

## UPDATE 17:45 — NEW LEADER and stricter target (supersedes the objective above)
New #1: **a9340e14-190b-473a-952f-65722b37d576** (0xZxh) "3-in-1 combo", **private 31.40%**, val 41.62%.
Exact edit (diffed vs baseline): PURE scaling (no mean term) e<-0.5e for 7 rows ' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm';
`</think>` row (id 248069) e<-1.3e; final norm (1+w)*1.4. Leader-equivalent spec (now job `r_newleader`):
`{"scale_tokens": {" wait":0.5,"wait":0.5," Wait":0.5,"Wait":0.5,"Hmm":0.5," hmm":0.5," Hmm":0.5}, "close_gain": 1.3, "gain": 1.4}`
apply_edit.py now also supports `scale_tokens` (pure scale, independent of `mode`), `close_gain` (scales the `</think>` row,
or the rows in `close_tokens`). Checkpoint for building on it: versions/a9340e14-190b-473a-952f-65722b37d576/model.
Owner's new target ("fortress", must survive a re-scoring on a larger hidden set — copies of the 3-in-1 within ±0.05 are useless):
private >= 34% (stretch 36%), validation >= 40.0%, one best submission + one ORTHOGONAL runner-up on the same parent stack.
Grid order requested: micro-grid of close_gain 1.15–1.50 with dampen+norm fixed, then stacked small nudges (filler set, mild rep-dir).
So: compare against **c_new** (the a9340e14 checkpoint) rather than c_leader; c_leader/c_base/c_think15/c_g1667 stay as calibration.
Queue order now: c_new, r_newleader, c_leader, c_base, c_think15, c_g1667, then agent jobs; c_g175/c_damp04/c_g125/c_rep run last.
The earlier queue failures were an orphaned vLLM process (fixed, killed by the orchestrator).

Correction (a6, verified by diff): c_g175 = 86968401 is (11 rows @0.6 mean, g=1.25), private 26.0. So g=1.25 vs 1.4 on the same rows: 26.0 vs 28.2.

## UPDATE 18:20 — use STRICT proxy accuracy (a6), not raw proxy acc
Raw proxy acc over-rewards `</think>` edits (c_think15: proxy 38.1 but private 17.6). Strict acc = correct rows that closed
normally (thinking >= 50 tokens AND not hitting max_tokens after the close) / all rows. Calibration so far (strict / raw / private):
c_think15 18.4 / 38.1 / 17.6; c_new 33.1 / 33.4 / 31.4; c_leader 25.3 / 25.3 / 28.2; c_base 8.1 / 8.1 / 18.6 (ordering holds).
Script: /home/user/delta/agents/a6/strict_score.py (needs results/NAME.json + NAME.texts.jsonl). Report strict acc for every
candidate, plus think_lt50 share and closed_overflow share. Also from a2: under a9340e14 Wait/Hmm are dead; truncations are
59% re-check prose cycles (If/So/But/This/Then/Maybe), 15% answer-phase overflow after close, 12% enumeration, 10% verbatim.

## UPDATE 18:30 — NEWEST LEADER 36.0% private: b8af1dbc-d52b-4bbc-a302-3505d897303c (xf3s), val 41.57
Parent a9340e14 + "little Qwen v0.1": (1) layer-16 MLP neuron 998 output column (down_proj[:, 998]) ×(−2);
(2) a unit "doubt" direction d (layer-13 residual, difference of means at thinking line starts: reconsider minus compute)
projected out of every o_proj / linear_attn.out_proj / mlp.down_proj in layers 0–13: W <- W − d(dᵀW).
Code + direction + write-up: /home/user/delta/xf3s/ (local) and ~/dl/xf3s/ (GPU box): build_edit.py, doubt_direction_L13.json, README.md.
Their findings: dose has a sharp peak at exactly ×1 projection (×0.5 and ×1.5 much worse); the edited model loops MORE when stuck
(verbatim grooves, can no longer escape via "Wait"); the published 15/19 rep edit aligns with a separate "stuck" direction (cos 0.37/0.46).
**New target: private > 36.0% (owner wants ≥ 36 + margin), val ≥ 40.** New calibration job `c_xf3s` runs first; compare against it.
To edit on top of it use `"base": "versions/b8af1dbc-d52b-4bbc-a302-3505d897303c/model"` in the spec — NOTE apply_edit then applies
scale_tokens/close_gain/gain MULTIPLICATIVELY on top of its already-edited rows/norm (e.g. close_gain 1.1 means 1.3×1.1 total).
Promising axes on this stack: escaping verbatim grooves (stuck/repetition direction, mild), answer-phase overflow after </think>,
re-tuning close_gain / g / damping given the doubt direction is gone, prose-mode re-check starters.

## Sensitivities (a3, teacher forcing) — size your steps accordingly
`</think>` logit at real close positions ≈ 30 (gain-1 units), so d(sampler logit)/d ln(close_gain) = g·z ≈ 42 nats: a 5% change
in close_gain moves the premature-close hazard more than any g in 1.0–2.0. Use close_gain steps of 1–3%, not 10%.
Pure-scale damping 0.5 is a kill switch (≈ −21 nats; c_new has 0 Wait/Hmm). Partial damping of other tokens needs s ≈ 0.94–0.98.

## A1 DECISION RULE (anchor c_xf3s; advisory, 6 calibration points)
private ≈ 12.84 + 0.566·strict (r 0.935, Spearman 0.94, LOO-RMSE 3.6). c_xf3s proxy: strict 37.8, short_think 1.2%, overflow 12.8%.
Paired SE of a strict difference on proxy400 at samples=1 ≈ 2.1–2.3 (≈1.25 private points).
(1) Screen vs c_xf3s: keep only if d_strict ≥ +3.5, short_think ≤ 3%, overflow ≤ 17%, and acc among healthy rows not down > 5 pts.
(2) Confirm on fresh seeds (prompts agents/a1/proxy_s2.json; reference job a1_xf3s_s2): mean d_strict over both seed sets ≥ +4.0.
(3) Never submit the argmax of a single-seed grid (winner's curse ≈ +3.5 strict for K=10).

## a4 notes (19:00)
- WARNING: the spec field `bias` is NOT an additive logit bias: h·mu/|mu|^2 has mean −38, sd 20 on a9340e14, so bias b on a row moves
  that logit by ≈ −38·b with large per-position variance. Don't use it as "+b".
- On c_xf3s, 33/41 answer-overflows are clean PREMATURE closes (not closes out of a groove); 0/41 overflow answers contain \boxed,
  so ending the answer phase earlier is worth ≈ 0. Raising close hazard further can only lose. a4 is fitting a premature-close
  direction for the `</think>` row (e_close −= β·u) and a think-start guard (p(</think>@t0) → ~1e-4) for b8af1dbc.

## UPDATE 19:20 — c092 FAILED confirmation; proxy noise is the binding constraint
a6_x_c092: seed set 1 strict 42.5 vs c_xf3s 37.8 (+4.7) but seed set 2 38.75 vs a1_xf3s_s2 44.1 (−5.3): pooled ≈ −0.3.
Unchanged xf3s differs by 6.3 strict between seed sets. Mechanism is consistent though: close 0.92 converts answer-overflow
(41/25 → 7/8 rows) into never-closed (60/69 → 115/116) with no net gain — the premature closes were on problems it cannot solve.
NEW EVALUATION PROTOCOL: reference job `ref_xf3s_x4` (b8af1dbc, proxy400, samples 4 = 1280 gens) runs first.
Evaluate candidates with `"samples": 3` (960 gens, ~10–15 min/job), compare with ref_xf3s_x4 (rows have `sample` 0..3; A1's
predict.py should pool). A candidate must show pooled d_strict ≥ ~+4 at samples≥3 to be a submission candidate.
Close-gain tweaks alone are now low priority; we need mechanisms that make UNSOLVED problems solvable or grooves escapable.

# ROUND 2 (20:45 UTC) — LAST SUBMISSION. Read this section; earlier sections are background.
Our submission 1 (b8af1dbc + published L15/L19 edit ×0.5) VERIFIED: private 33.0, val 41.44, trunc 0.15 (parent 36.0/41.57/0.23).
Lesson: the server rewards ACCURACY OF FINISHED ANSWERS, not finishing. Our local acc|healthy dropped 5 points and that was the warning.
Leaderboard now: #1 b8af1dbc 36.0 (xf3s); xf3s's own gentler dose (doubt ×0.85) d6b5f7e3 = 33.2; mihaikoodo 632210ae
(doubt + n998 + hesitation 0.5 + norm 1.4 from base) = 33.8 → 36.0 likely contains ~+2 luck (winner's curse).
A1 rule (updated): vs pooled xf3s reference (ref_xf3s_x4 + c_xf3s + a1_xf3s_s2, strict 42.3): pooled d_strict ≥ 4.5, z ≥ 2.5,
predicted private ≥ 38.3, guards short ≤ 3%, overflow ≤ 17%, acc|healthy drop ≤ 5 — treat acc|healthy as a HARD filter now.
Tool: `python /home/user/delta/agents/a1/predict.py CAND.json [CAND_s2.json]` (needs .texts.jsonl next to each json).
FORBIDDEN: any `</think>` scale; full-strength published rep edit; L15/L19 rep ×0.5 again; copies of 3-in-1 without a new axis.
Orchestrator already queued (samples 3): v2_dx15 / v2_dx17 (xf3s doubt projection extended to layers 14–15 / 14–17),
v2_g105 / v2_g110 (final-norm gain ×1.05 / ×1.10 on b8af1dbc). Builder: ~/dl/build_doubt_ext.py.
Round-2 agents: max 2 GPU jobs each, always "samples": 3, names r2<tag>_*, report within ~60 minutes.
