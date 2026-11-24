# A2 — token-set geometry (saved by orchestrator from the agent's hand-back; proxy advisory, TF first-order)

- Row edits: pure scale Δz = −(1−s)·z with competitive z ≈ 25–40: s 0.98 −0.7, 0.95 −1.3..−1.7, 0.90 −2.5..−3.5,
  0.5–0.6 −12..−17 nats (kill). Mean mode 0.6 ≈ pure 0.5 (mean term ≈ +1.2 nats). Partial damping only for s ≈ 0.93–0.99.
- Killed Wait mass moves to Let/If/So/But, never to `</think>` (base traces: ' Wait' 279 → 0; ' If' 298 → 332; `</think>` 0.8 → 0.75).
- Over-representation in truncated tails (bench-controlled): c_new Then 5.7, If 3.0, But 2.9, This 2.6, Maybe 2.2, So 2.0;
  c_xf3s Maybe 4.2, Let's check 4.1, But 2.6. Actually/Okay/Final mark successful traces. Collateral in final answers:
  This 40%, Let 29%, If/So 20%, But 8.8%, Then 7.1%, Maybe 4.7%.
- c_new: in 49/79 truncated math traces the gold answer appeared at median 17% of the trace; truncated traces are prose-mode from
  the start (bullet-line share 0.53 vs 0.83).
- Close decision points (p>0.02) are rare (0–2/trace), at line starts after wrap-up lines; competitor = indentation tokens.
  p(close@token0): 0 for c ≤ 1.3, 0.05–0.07 at 1.35, 0.06–0.19 at 1.40, 0.20–0.69 at 1.50.
- Truncation taxonomy (base / leader / c_new / c_xf3s): re-check 95.5/58.4/58.8/3.0%; answer overflow 1.4/0.9/13.6(+1.7)/40.6%;
  verbatim groove 0.7/16.0/9.6/32.7%; enumeration 1.0/14.2/12.4/20.8%.
- b8af1dbc: grooves are top-p locked (p(top1)=1.000 at 10th pct); restoring Wait rows is inert; answer loops have no
  <|im_end|> decision point and no \boxed.
- Close margins (93 c_xf3s traces): overflow −2.3/−0.6/+0.7 (p 0.34), correct −1.0/+2.0/+10.6 (p 0.72), wrong −2.3/0.0/+4.8.
  Share kept at that token (overflow/correct/wrong): f .97 .39/.73/.52; .95 .20/.60/.34; .92 .05/.46/.19; 1.03 ×1.84/×1.33/×1.65.
- GPU jobs: 0 used (three withdrawn).

## Recommendation
1. No more hedge-row kills on b8af1dbc. 2. If any close tweak: f ≈ 0.95–0.97 rather than 0.92 (note: a6_x_c092 pooled ≈ 0).
3. Optional runner-up (untested): scale_tokens {" Maybe":0.95,"Maybe":0.95," But":0.97,"But":0.97} on b8af1dbc.
4. Grooves (33%) and enumeration (21%) need representation-level or global changes.
