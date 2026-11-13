# Delta League sprint — handoff (05.11.2026, ~21:00 UTC)

Owner: SpacexCeo (RAEN). Competition `qwen35-08b-v1` (Qwen3.5-0.8B), config `public-ladder500-8k-v2`. Rank = PRIVATE test only.

## UPDATE 06.11 07:30 UTC — compute changed
Boxes 1–4 (5090, 4090, 2× 5070 Ti) expired overnight: their data is gone, including the round-2 queue (v2_g105, v2_dx17,
v2_g110, r2b_dig115, r2b_dl115, r2a_a1, r2a_a05 — never scored) and the remote checkpoints/generations. Only v2_dx15 was scored (REJECT).
New box 5: 2× RTX 5070 Ti 16 GB, Jupyter at / (token auth, credentials with the owner). Ready: ~/venv (vLLM), ~/dl with
b8af1dbc checkpoint, prompt sets (proxy400, proxy_s2, stress), xf3s direction, upstream edit, runner.py running on GPU 0,
and the full handoff archive ~/sprint_handoff.tar.gz (sha256 56940f08…2603). The xf3s reference (ref_xf3s_x4 etc.) must be
re-run on the new hardware before comparing new candidates (vLLM results depend on hardware/batching).

## State at handoff
- Leaderboard #1: `b8af1dbc-d52b-4bbc-a302-3505d897303c` (xf3s, a9340e14 + doubt-direction projection L0–13 + L16 n998 ×−2),
  private 36.0 / val 41.57 / trunc 0.23. Its own gentler dose (doubt ×0.85, d6b5f7e3) = 33.2; mihaikoodo 632210ae = 33.8 →
  36.0 likely contains ~+2 of luck (winner's curse).
- Our submission 1: `51721d34-391e-4c58-bc5d-57bccff5016e` → version `344c8376-66a1-413b-abca-da2b5b986044` =
  b8af1dbc + published L15/L19 repetition edit ×0.5 → **private 33.0 / val 41.44 / trunc 0.15**, rank 4.
  Research PR: https://github.com/ilblackdragon/deltaleague/pull/24 (fork SpacexCeo/deltaleague, branch claude/compassionate-albattani-fdhmk0).
- Quota: daily_limit 2, used 1 today (`delta quota`). Owner calls the remaining one "the last shot". CLI is logged in as
  SpacexCeo in the old sandbox only; a new environment needs `delta login --no-browser` + owner approval.
- No candidate has passed the A1 rule. Do NOT submit unless one does.

## The decision rule (A1, agents/a1/REPORT.md, predict.py, calib.json)
Strict accuracy = correct AND closed `</think>` AND ≥ 50 thinking tokens AND not truncated, over all rows.
private ≈ 13.33 + 0.507·strict (n=10 leaderboard checkpoints, r 0.94, LOO-RMSE 2.5). Reference = b8af1dbc pooled over 6 draws
(ref_xf3s_x4 [4 samples] + c_xf3s + a1_xf3s_s2): strict 42.3, acc|healthy 60.1.
SUBMIT only if: pooled d_strict ≥ 4.5 over ≥ 2 seed sets at samples ≥ 3, z ≥ 2.5, predicted private ≥ 38.3, val ≥ 40,
short-think ≤ 3%, overflow ≤ 17%, **acc|healthy drop ≤ 5 (HARD filter — submission 1 failed exactly this way)**.
Noise: single-seed strict SE ≈ 2.7–4; same seeds do NOT reproduce across runs with different `samples` (vLLM batch effects).

## Key findings (details in agents/*/REPORT.md and REPORT.md)
1. Server private tracks finishing within 8,192 tokens only up to a point; on the xf3s stack it rewards ACCURACY OF FINISHED
   answers. Edits that raise the closed share but lower acc|healthy lose on private (sub 1: trunc 0.23→0.15, private 36→33).
2. Head knobs are exhausted on b8af1dbc: `</think>` row scale (×0.92: overflow→never-closed one-for-one, pooled ≈0; ×0.96 −3.3),
   final-norm gain ×0.9/0.95 neutral, think-start guard −1.4, premature-close direction −6.4, Wait/Hmm damping redundant
   (n998 + projection already kill Wait). close_gain sensitivity ≈ 42 nats per ln unit; s ≤ 0.6 damping is a kill switch.
3. b8af1dbc truncations: 41% answer-phase overflow after a marginal early close, 33% verbatim grooves (top-p locked: p(top1)=1.000),
   21% enumeration, 3% re-check. Overflow answers never contain \boxed (ending them earlier scores 0).
4. Published L15/L19 edit is rank-1 per layer, its gate fires ~4× more at groove line starts (conditional groove escape, A5);
   it rescues never-closed rows but most rescued rows end wrong; ×1.0 breaks MMLU letters; ×0.5 = submission 1.
5. R2B: the acc|healthy drop under sub 1 was mostly mix (hard rows newly finished) + a real GSM8K per-item loss (75.5→71.0)
   from premature commitment (wrong answers ~26% shorter thinking). Answer-format misses on b8af1dbc are rare (≤0.6 strict).

## Round 2 jobs (box 1 queue, samples 3 unless noted) — state at handoff
| job | edit (on b8af1dbc) | status |
|---|---|---|
| v2_dx15 | doubt projection extended to layers 14–15 | done: d_strict −0.79 ± 1.21 → REJECT |
| v2_g105 | final-norm gain ×1.05 | running at handoff |
| v2_dx17 | doubt projection extended to layers 14–17 | queued |
| v2_g110 | final-norm gain ×1.10 | queued |
| r2b_dig115 | digit rows '0'..'9' group-sharpened s=1.15 (R2B) | queued |
| r2b_dl115 | digits + letters 'A'..'J' s=1.15 (R2B) | queued |
| r2a_a1, r2a_a05 | R2A (Fable) residual-doubt candidates, agent stopped mid-work — check ~/dl/agents/r2a/ | queued |
Score each: `python tools/q.py result NAME results --texts && python agents/a1/predict.py results/NAME.json`. A passer needs a
`_s2` run (prompts agents/a1/proxy_s2.json) and a stress run (prompts stress.json, compare with s_xf3s) before submission.

## Compute (servers stay rented — owner decision; credentials are with the owner, never in this package)
- Box 1: RTX 5090 32 GB, Jupyter at /jupyter (HTTP basic auth). Holds everything: ~/dl (versions/ checkpoints incl. all
  calibration leaders, results/ with full generations, agents/ work dirs, queue/ done/ failed/, runner.py running).
- Box 2: RTX 4090 24 GB; Box 3 and Box 4: RTX 5070 Ti 16 GB. Jupyter at / (token auth). vLLM env in ~/venv, b8af1dbc in
  ~/dl/versions, prompt sets and runner.py (idle, empty queue) ready — use them for parallel _s2/stress runs.
- Access from a sandbox without SSH: tools/gpu.py over the Jupyter REST/websocket API. Put one JSON per box in the dir given by
  $GPU_BOX_DIR: `.box1.json` = {"host": "...", "base": "/jupyter", "basic_file": "<file with user:password>"},
  `.boxN.json` = {"host": "...", "base": "", "token": "..."}; select with `BOX=N`. q.py submits/reads jobs (BOX env respected).
- Queue job spec: see AGENT_BRIEF.md "Infrastructure" (apply_edit.py fields: tokens/mode, scale_tokens, close_gain, gain, bias
  [do not use — position-dependent], rep_alpha/rep_dir) or {"model": "<dir>"}; "samples", "prompts".
- Submitting: build with `delta patch build --parent versions/<parent>/model --modified <cand>/model.safetensors --output sub`,
  `delta validate sub`, open the research PR first (fork SpacexCeo/deltaleague; the owner clicks "Create PR" upstream), then
  `delta submit sub --title ... --pr <url>`; watch with tools/watch_sub.sh <submission id>.

## Package contents
REPORT.md (owner summary, RU), AGENT_BRIEF.md (all facts/updates in order), all_jobs_summary.csv (33+ jobs: acc, trunc, strict,
regimes, per-bench), agents/<a1..a6,f1,r2a,r2b>/ (REPORT.md + all code + small data), results/ (proxy results + generations
fetched to the sandbox), gpu/ (eval_vllm.py, apply_edit.py, runner.py, build_*.py, setup_box.sh, summarize_all.py), tools/
(gpu.py, q.py, watch_sub.sh), xf3s/ (leader's public research record), specs/, pr/ (PR body), lb*.json (leaderboard snapshots).
Excluded: model weights, teacher-forcing tensor caches (*.pt/*.npz/*.npy, regenerable with the agents' scripts), credentials.
