# A4 — termination dynamics (saved by orchestrator from the agent's hand-back; advisory proxy numbers)

- Truncated traces mostly do NOT hold the answer: never-closed traces stating the gold answer: c_base 23.8%, c_leader 33.2%,
  c_new 18.7%; forced-close probe on c_new never-closed traces correct at any point in 2/12.
- Closing on c_new is a switch: at real closes p(close) median 0.91 (logit ≈ 39); elsewhere the nucleus removes it (summed p over
  8192 tokens in never-closed traces median 0.000). `</think>` logit ≈ 27–30 at line starts, ≈ 0 elsewhere → row scaling is a
  newline-only bias (c−1)·30 nats. Min line-start gap ≈ 5 nats in never-closed vs 4.7 in closed traces → any boost that closes
  stuck traces also closes good ones early. Hazard model predictions (c_new): close 1.35 −5 raw, 1.40 −11 raw, 1.50 collapse.
- Conditional directions on c_new (2-fold CV): "answer-known" trunc 0.455→0.414 with no acc change; "close-ready" and
  "soft budget" lose; think-start guard works.
- `bias` field: h·mu/|mu|² mean −38, sd 20 → moves the logit by ≈ −38·b, position-dependent. Do not use on `</think>`.
- b8af1dbc: 41 overflow rows on c_xf3s = 30 answer-phase verbatim loops, 8 long answers, 3 late closes; 33/41 clean premature
  closes; 0/41 contain \boxed; truncating answers scores 0/41. c092 transitions: overflow→never 28, overflow→normal 9 (4 correct),
  normal→never 45, normal→normal 171 (122 correct).
- Premature-close direction u (41 overflow vs 47 normal close states): held-out AUC 0.84. Survival of natural close
  (overflow / normal / correct-normal / extra early close): ×0.97: .35/.55/.60/0; ×0.92: .04/.26/.34/0; ×0.97−2u: .31/.72/.78/.08;
  ×0.96−2u: .23/.67/.74/.04 → ≈1.6× more normal closes kept at equal premature suppression.
- Think-start guard: ridge readout w of "first generated position" (held-out ≥0.976 at t=0, ≤0.21 elsewhere); e_close −= 6w →
  p(close@t0) 0.018/0.062 (mean/max) → 4.7e-5/1.6e-4 on a9340e14; KL thinking 3e-5, first 16 answer tokens 2.4e-3.
- Candidates: cand_xg6 = b8af1dbc, e_close −= 6·gx_w (sha256 f953e07f…), job a4_xg6; cand_xpd = b8af1dbc,
  e_close ← 0.97·e_close − (2·prem_u + 6·gx_w) (sha256 de2a189d…), job a4_xpd. Builder: build.py PARENT OUT U.npy BETA [C].

## Recommendation
1. Think-start guard: a safe stack component (expected ≈ +0.5 strict, invisible at current noise), accept if short_think ≈ 0 and
   normal-row accuracy not down vs ref_xf3s_x4.
2. Premature-close direction (a4_xpd): the only edit targeting overflow selectively; unproven — re-run at samples 3 only if the
   single seed shows d_strict ≥ +3.5 with never-closed not rising by the same amount.
3. No more queue on raising close hazard. 4. Untested: an "answer-known" close direction on lower-close stacks.
