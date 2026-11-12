# Proxy B calibration (2026-11-06)

Proxy B: 460 items (proxyb/build_proxyb.py), 2 samples, server sampler, 8192 tokens.
Run on 14 board versions; private scores from proxyb/calib_versions.txt. Full output: calib_output.txt.

## Findings
- Overall proxy-B accuracy vs private: r = 0.62 (Spearman 0.58).
- Strongest per-bench predictors of private: GSM8K r=+0.88, BBH +0.87, MATH-500 +0.80, Winogrande +0.69, MMLU +0.56.
  CSQA +0.41, PIQA +0.37; ARC-E, BoolQ, OBQA, HumanEval ~0; IFEval -0.43.
- This reverses the earlier inference from the board's validation columns (math anti-correlated): on our own
  generations with the server sampler, reasoning benches track private best across the whole range.
- Leave-one-out composite (z-average of benches with train r>0.5, linear fit): LOO r=0.76, RMSE 2.2 private points
  (sd of private 3.35). Private itself moves ~2 points on re-evaluation (344c8376: 33.0 -> 35.0), so the top cluster
  (35.0 / 34.8 / 34.2 / 32.8 / 32.4) is not separable by any local proxy calibrated on board labels.
- LoRA v3 (6c2acb28) has the highest composite (pred 34.1-34.5) but private 32.4: residual -2.1, within noise,
  but it is the one candidate the proxy was meant to rank and it ranked it wrong.

## Consequence
Only a large true gain (>= 3 private points, i.e. >= ~1 composite z-unit) can be confirmed locally.
Edits predicted at +1 are indistinguishable from noise before submission.
