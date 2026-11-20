# A5 — re-check direction, the published 15/19 edit, groove escape on b8af1dbc (saved by orchestrator; advisory)

- Re-check vs progress direction on a9340e14 (32 traces): split-half cos 0.90–0.99, held-out AUC 0.78–0.86 (1-D); lens + reconsider/
  revisit/check, − So/Thus/Then. Same mechanism as xf3s's d13 (cos 0.30 at L13). Full ablation too strong (KL 0.14).
- Published 15/19 edit is exactly rank-1 per layer (s·u·vᵀ, v on 64 neurons, cos(u15,u19)=0.85); u ⟂ re-check/close/d13
  (|cos|<0.08), logit lens = "commit" tokens. On b8af1dbc, 40/60 truncations are grooves; u anti-aligns with the groove direction
  (cos −0.35..−0.53, L15–21) and the stuck direction (−0.54..−0.71); its gate fires ~4× more at groove line starts → conditional
  groove escape. Offline ×1 selective (KL 0.015), ×2 not (0.056). L15 carries the groove effect; L19 pushes `</think>` at closes.

| job | d_strict vs ref | trunc | never-closed | acc|normal |
|---|---|---|---|---|
| a5_xrep10 (×1 both layers, 3 samples) | +2.08 ± 1.63 | 28.5% → 16.0% | 18.8% → 8.2% | 59.5 → 53.1 |
| a5_xrep10_s2 (fresh seeds) | +0.4 | 29.4% → 15.9% | 21.6% → 10.3% | |
| a5_xup15 (layer 15 only, 3 samples) | +1.88 ± 1.62 | 28.5% → 16.4% | 18.8% → 8.7% | |

xrep10 transitions: 101 never-closed → wrong, 34 → correct. Pooled both-layer ≈ +1.25 strict (≈ +0.7..1.1 private).

## Recommendation
Nothing passes the A1 rule. Orthogonal runner-up if wanted: b8af1dbc + published edit on layer 15 only (agents/a5/cand_up15,
built bit-exact by ~/dl/agents/a5/build_up.py). Don't raise the dose. Next: pair groove escape with something that improves what
happens after escaping (rescued loops mostly end wrong).
(Orchestrator note: the submitted candidate is the half-strength both-layer version, α=0.5, which pooled +2.4 ± 1.2 strict over
two seed sets and passed the stress set.)
