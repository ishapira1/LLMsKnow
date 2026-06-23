# Friction Hypothesis Summary

This note has been updated after the June 20, 2026 rerun. Older pooled quartile and correlation numbers were removed because they came from the pre-rerun analysis.

For current numbers, use `current_empirical_summary_2026-06-20.md`.

Current paper-safe friction claim:

Neutral confidence acts like friction. In the June 20 rerun, high-confidence neutral-correct examples are harder to move under `incorrect_suggestion_strong` than neutral-correct examples overall.

Current anchor numbers from the June 20 summary:

| subset | external wrong after strong bias | wrong cases toward `b` | random_all probe correct after bias |
|---|---:|---:|---:|
| all neutral-correct | 42.3% | 93.8% | 80.7% |
| high-confidence neutral-correct | 23.9% | 97.0% | 93.9% |

Interpretation:

- Confident correct answers are harder to move.
- When confident examples do move, the error is still targeted toward the user-backed wrong answer `b`.
- The `random_all` probe preserves a strong correct-answer signal in the high-confidence subset.

Do not reuse the old quartile movement numbers from the retired pre-rerun analysis.
