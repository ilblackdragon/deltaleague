# A1 — calibration and decision statistics (saved by orchestrator from the agent's hand-back; advisory)

Calibration points (private): c_xf3s 36.0, c_new 31.4, c_leader 28.2, c_base 18.6, c_think15 17.6, c_g1667 17.2.

| job | private | server trunc | acc | closed% | acc|closed | tokens | overflow% | short% | acc|healthy | strict |
|---|---|---|---|---|---|---|---|---|---|---|
| c_xf3s | 36.0 | ? | 38.8 | 68.4 | 47.7 | 3726 | 12.8 | 1.2 | 56.3 | 37.8 |
| a1_xf3s_s2 (fresh seeds) | | | 44.7 | 70.6 | | 3643 | 7.8 | 0.6 | 62.9 | 44.1 |
| c_new | 31.4 | ? | 33.4 | 44.7 | 62.9 | 6052 | 8.4 | 1.2 | 74.6 | 33.1 |
| a1_new_s2 (fresh seeds) | | | 35.6 | 44.4 | | 6091 | 10.6 | 1.9 | 78.9 | 35.0 |
| c_leader | 28.2 | .526 | 25.3 | 31.6 | 78.6 | 6970 | 0.6 | 0 | 80.2 | 25.3 |
| c_base | 18.6 | .746 | 8.1 | 9.4 | 76.5 | 7899 | 1.2 | 0 | 86.7 | 8.1 |
| c_think15 | 17.6 | .440 | 38.1 | 56.2 | 39.2 | 4955 | 40.9 | 49.1 | 61.5 | 18.4 |
| c_g1667 | 17.2 | .826 | 4.4 | 4.7 | 93.3 | 7940 | 0 | 0 | 93.3 | 4.4 |

Fits: UPDATED n=10: private ≈ 13.33 + 0.507·strict, r 0.939, Spearman 0.95 [0.80,1.00], LOO-RMSE 2.5 (earlier n=6: 12.84 + 0.566·strict, LOO 3.6).
Raw acc LOO 8.6; acc|closed r −0.32 (useless). Server-only: private ≈ 8.72 + 36.67·(1−trunc) (LOO 2.46, excl. think-forcing);
validation does not predict private (Spearman −0.09). Server closed% ≈ 17.7 + 0.74·proxy closed% (n=4).

Noise: per-run SE 2.4–2.8; fresh-seed replicate of unchanged xf3s +6.25 strict (paired SE 2.7); no common-random-number benefit
(edits diverge at once). Paired SE of d_strict: 1v1 2.71 (z≥2.5 needs 6.8); 3 vs 4 samples 1.46 (needs 3.65 ≈ +2.1 private).
Best-of-K inflation on one seed set: K=3 +1.9, K=5 +2.6, K=10 +3.4 strict.

## Decision rule
1. Guards: short-think ≤ 3%, overflow ≤ 17%, acc|healthy drop ≤ 5. 2. Screen (1 sample) d ≥ 3.5 → CONFIRM only.
3. Confirm with ≥2 samples/prompt vs pooled xf3s reference: d ≥ 4.5 (updated), z ≥ 2.5, predicted private ≥ 38.3.
   Note: equal seeds do not reproduce across runs with different `samples` (vLLM batch dependence); every run counts as an
   independent draw. xf3s reference pooled over 5 draws: strict 41.9 (draws 43.4/42.5/40.3/45.6/37.8).
4. Submit the single best pooled candidate; validation ≥ 40. Worked example a6_x_c092: +4.69 / −5.31 → pooled −0.31 ± 1.86 REJECT.
Warnings: winner's curse on 36.0; n=6; proxy mix ≠ private mix. Tools: predict.py, analyze.py, calib.json, fetch.sh.
