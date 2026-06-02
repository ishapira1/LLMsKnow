# Friction hypothesis summary

Data used: 4 paired runs, clean test subset only, 6,379 items total and 5,065 neutral-correct items.

Key claim supported:
Under the same incorrect-suggestion intervention, weakly committed items move much more toward the user-backed wrong answer b. This remains true even when items do not flip, and even when they stay correct.

Main pooled numbers:

- All neutral-correct items, confidence = P(c)-P(b):
  - ΔP(b) by quartile: 21.0, 9.8, 3.4, 0.8 points
  - gap closure by quartile: 52.0, 28.3, 10.9, 2.5 points

- Neutral-correct items that do not flip to b, confidence = P(c)-P(b):
  - ΔP(b) by quartile: 2.44, 2.29, 1.39, 0.36 points
  - gap closure by quartile: 19.8, 14.0, 7.0, 1.5 points

- Neutral-correct items that stay correct, confidence = P(c)-P(b):
  - ΔP(b) by quartile: 3.39, 2.41, 1.38, 0.36 points
  - gap closure by quartile: 7.51, 7.77, 4.46, 1.14 points

Robustness to confidence definition on the stay-correct subset:

- Confidence = P(c)-P(best wrong):
  - ΔP(b): 3.16, 2.39, 1.49, 0.40 points

- Confidence = S(c)-S(b):
  - ΔP(b): 3.12, 2.17, 1.48, 0.47 points
  - same-probe gap shrink: 0.236, 0.375, 0.238, 0.107

Run-by-run endpoint comparison on the stay-correct subset, confidence = P(c)-P(b):

- Llama ARC: ΔP(b) 11.95 vs 1.10 points, gap closure 21.69 vs 2.38 points
- Qwen ARC: ΔP(b) 1.59 vs 0.12 points, gap closure 3.85 vs 0.25 points
- Llama CSQA: ΔP(b) 2.72 vs 0.28 points, gap closure 10.30 vs 1.80 points
- Qwen CSQA: ΔP(b) 2.43 vs 0.15 points, gap closure 4.04 vs 0.30 points

Spearman correlations on the stay-correct subset:

- Output confidence P(c)-P(b) vs ΔP(b):
  - Llama ARC: -0.393
  - Qwen ARC: -0.172
  - Llama CSQA: -0.379
  - Qwen CSQA: -0.114

- Probe confidence S(c)-S(b) vs ΔP(b):
  - Llama ARC: -0.328
  - Qwen ARC: -0.305
  - Llama CSQA: -0.333
  - Qwen CSQA: -0.217

What this supports:
The friction story is real on the probability scale. It is not just that weaker items are closer to the flip boundary. They actually move more toward b under the same intervention.

What this does not support:
It does not by itself show that every item experiences the same additive evidence shift in logit space. The strongest supported statement is about probability movement and gap closure.
