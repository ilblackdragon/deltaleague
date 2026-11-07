# A6 — head-knob hillclimb (saved by orchestrator from the agent's hand-back; proxy numbers advisory)

Verdict: no head-knob nudge beats b8af1dbc (36.0).

| spec (relative to b8af1dbc) | strict s1 | d s1 | strict s2 | d s2 | mean d |
|---|---|---|---|---|---|
| xf3s (c_xf3s / a1_xf3s_s2) | 37.8 | 0 | 44.1 | 0 | 0 |
| close ×0.92 (a6_x_c092 / _s2) | 42.5 | +4.7 | 38.75 | −5.3 | −0.3 rejected |
| g ×0.95 (a6_x_g095) | 39.7 | +1.9 | – | – | fails screen |

- Leaderboard surface over (s,g) for the old 11-row family is unidentifiable (5 points, 5 params, all but b0 within 1 SE of 0);
  only a positive damping×gain interaction is suggested.
- Iso-suppression: the relative (softmax-gap) rule holds, ds/dg > 0. Damping is a saturated kill switch: any s ≤ 0.7 gives 0
  Wait/Hmm per 1k. On b8af1dbc the Wait/Hmm damping is redundant (0.012 per 1k fully undamped).
- Live head knob is the `</think>` gain c. On c_new traces p(close@pos0): c 1.2→0, 1.3→0.013, 1.4→0.16, 1.5→0.53; predicted skip
  matches measured (c_new 1.3 vs 1.25%, c_think15 50 vs 49%). On c_xf3s (relative): premature closes /1k: none 0.214, ×0.92 0.003,
  ×0.95 0.034, ×0.97 0.072, ×1.03 0.59; natural-end acceptance 0.68 / 0.28 / 0.41 / 0.52 / 0.83. g ×0.95/×1.05: 0.261/0.172.
- Regimes (never / normal(acc) / overflow / skip): xf3s s1 60/215(56.3)/41/4; c092 s1 115/198(68.7)/7/0; xf3s s2 69/224(62.9)/25/2;
  c092 s2 116/196(63.3)/8/0 → lowering c converts overflow into never-closed one-for-one. Single-seed strict SE ≈ 3.5–4.
- Calibration: strict vs private: c_think15 18.4/17.6, c_base 8.1/18.6, c_leader 25.3/28.2, c_new 33.1/31.4, c_xf3s 37.8/36.0.

## Recommendation
1. Keep b8af1dbc's head knobs. 2. At most close ×0.97 (expected ≈ 0 ± 2 private). 3. Orthogonal idea: gain ×1.05 (not run).
Remaining gains need context-dependent closing (a4 direction) or loop-escape edits, not (s, g, c).
Files: grid.csv, strict_score.py, tf_hazard.py, eval_hazard.py, fit_surface.py, rowdiff.py, grid_run.sh, apply_edit_local.py.
