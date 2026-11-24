# R2B — accuracy of finished answers on b8af1dbc (saved by orchestrator; advisory)
- The acc|healthy drop of our submission (60.1 → 55.7) is mostly a mix effect (newly finished rows are hard); on normally-closed
  prompts strict did not fall (60.5 → 61.3). Real per-item loss: GSM8K 75.5 → 71.0 on normally-closed prompts.
- Mechanism: premature commitment — under the edit wrong answers have ~26% shorter thinking than correct ones on the same prompt
  (not significant on b8af1dbc); the answer phase re-derives and is 10–20 points less accurate.
- Answer-format misses are too rare on b8af1dbc to be a lever (≤ 0.6 strict).
- Candidates (queued on box 1, results pending at handoff): r2b_dig115 = digit rows '0'..'9' group-sharpened s=1.15
  (e_t ← m_G + s(e_t − m_G)); r2b_dl115 = digits + letters 'A'..'J' s=1.15. Builder ~/dl/agents/r2b/build.py (box 1).
  Expected gain 0–2 strict (≈ paired SE). Pass condition: acc|healthy ≥ ~60, healthy share ≥ ~70%, d_strict > 0, then _s2.
