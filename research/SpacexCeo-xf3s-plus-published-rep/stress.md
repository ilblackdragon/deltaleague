# Stress set (local, advisory)
265 public prompts, 1 sample, same sampler as the competition config (`stress_build.py`, scorer in `eval_vllm.py`):
GSM8K 40, MATH-500 40, MMLU 40, BBH 40, IFEval 40 (lm-eval instruction checker), HumanEval 30, MBPP 30 (tests executed), plus 5
fixed free-form prompts checked for normal `</think>` closing and hedge loops.

| bench (n) | parent b8af1dbc | α = 1.0 | α = 0.5 (submitted) |
|---|---|---|---|
| all (260 scored) | 40.0 | 38.1 | 45.3 |
| GSM8K (40) | 57.5 | 57.5 | 52.5 |
| MATH-500 (40) | 25.0 | 32.5 | 37.5 |
| MMLU (40) | 50.0 | 27.5 | 40.0 |
| BBH (40) | 37.5 | 35.0 | 62.5 |
| IFEval (40) | 45.0 | 52.5 | 60.0 |
| HumanEval (30) | 33.3 | 33.3 | 30.0 |
| MBPP (30) | 33.3 | 30.0 | 33.3 |
| truncated | 0.18 | 0.07 | 0.06 |

α = 1.0 often closed early and answered MMLU in prose without an option letter (12 parent-correct items lost vs 3 gained),
so it was rejected. Fixed prompts: all 5 close normally with 0 hedge words for α = 1.0; each bench row has n = 30–40, so
single-bench differences of ≤ 10 points are within noise.
