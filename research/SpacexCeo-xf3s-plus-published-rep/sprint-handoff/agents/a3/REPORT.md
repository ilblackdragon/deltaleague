# A3 — sampler maths: gain g × presence penalty 1.5 × top-k 20 / top-p 0.95 (saved by orchestrator; proxy advisory)

Sampler logit ℓ_v = g·r_v·z_v − 1.5·[v used]. A pure row scale r shifts that token's absolute logit by g(r−1)z_v (z ≈ 25–36 for
top tokens): close_gain sensitivity d ℓ/d ln c = g·z_T ≈ 42 nats; pure-scale 0.5 ≈ −21 nats (kill switch); partial damping needs
s ≈ 0.94–0.98 (0.97 halves Maybe/Actually/Alternatively mass, 0.94 cuts it to 1/5).
Close-hazard elasticities: d lnΣ/d ln c = +25 (truncated) / +12 (correct) on c_new, +42 / +33 on xf3s; d lnΣ/d ln g ≈ −1.8 / −0.9
→ g and close_gain nearly orthogonal (−7% g ≈ close ×1.005).
Presence penalty saturates (used-token nucleus mass 0.77 → 0.97 over the trace); late it is a uniform shift + 1.5 nat bonus for
unused tokens, i.e. +1.5/g. `</think>` is always unused, so the penalty is its main ally (no penalty: close hazard 0.99→0.66
correct, 0.36→0.17 truncated). Penalty does not hurt maths (digits Δlog q −0.0016). Per-token temperatures: not worth it.
g 1.0→1.4→2.0: entropy 0.46→0.30→0.20 (accuracy benefit); groove escape per token 0.033→0.022→0.017 (∝1/g); unclosed truncations
ending in a verbatim groove 0.18 (g1.3) / 0.20 (1.4) / 0.41 (1.55). In base, sharpening raises Wait mass (argmax) and lowers
p(</think>) at closes → why g=1.667 alone hurts; with damped rows sharpening suppresses remaining hedges (old leader's synergy).

| job | stack | Δ vs ref | notes |
|---|---|---|---|
| a3_new_g13 (g 1.3) | a9340e14 | +2.2 ± 2.1 raw | trunc .537 vs .553 |
| a3_new_g155 (g 1.55) | a9340e14 | −0.6 ± 2.0 | grooves double |
| a3_x_g090c992 (g×0.9, close×0.992, samples 3) | xf3s | pooled strict +0.68 ± 1.44 | REJECT; grooves 0.120→0.084/gen, never-closed unchanged |

## Recommendation
No submission from A3. Optional low-risk add-on: {"base": b8af1dbc, "gain": 0.9, "close_gain": 0.992} (≈ +0–1 private, within noise).
Don't: g > 1.4, per-class temperatures, lower g + higher close. What could still move the score: what happens after a groove breaks.
