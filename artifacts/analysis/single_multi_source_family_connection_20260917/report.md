# Single-turn vs multi-turn matched-cohort analysis

## Design and audit

- Exact matched responses: 3,888.
- Primary matched question-suggestion cells: 1,296 (three repetitions per placement).
- Questions: 81.
- Matching fields: model, question, endorsed wrong answer, source family, template, and repetition.
- Primary analysis aggregates the three exchangeable API repetitions before comparing placements.

## Main result

Averaged with equal weight over models, questions, and source families, multi-turn adoption is -9.5 percentage points relative to single-turn (95% question-cluster bootstrap CI [-12.6, -6.7]). This pooled effect is not a general law: model-by-source placement effects cross zero and reverse direction.
Single-turn behavior is informative about multi-turn behavior, but it is far from interchangeable. Across the 1,296 matched model-question-source cells, the raw adoption-rate correlation is 0.49 [0.44, 0.53].
After removing each model-by-source-family mean in both placements, the correlation is 0.43 [0.37, 0.49]. This residual association asks whether unusually susceptible questions in single-turn are also unusually susceptible in multi-turn, beyond model and source-family baselines.
Three-draw repeatability is alpha=0.97 in single-turn and alpha=0.95 in multi-turn. The attenuation-corrected group-rate correlation is 0.51.

## Placement effects by model and source family

| Model | Source family | Single | Multi | Multi - single (pp) | 95% cluster bootstrap CI | BH q |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.6 Terra | Unsupported user | 18.1% | 47.3% | +29.2 | [+20.2, +38.7] | 0.0000 |
| GPT-5.6 Terra | Individual expert | 85.6% | 72.0% | -13.6 | [-20.6, -7.0] | 0.0003 |
| GPT-5.6 Terra | Authoritative reference | 88.1% | 84.8% | -3.3 | [-7.4, +0.4] | 0.2168 |
| GPT-5.6 Terra | Independent corroboration | 88.5% | 74.5% | -14.0 | [-22.6, -5.8] | 0.0030 |
| GPT-5.6 Luna | Unsupported user | 31.3% | 44.4% | +13.2 | [+5.3, +21.0] | 0.0028 |
| GPT-5.6 Luna | Individual expert | 86.8% | 62.6% | -24.3 | [-33.3, -16.0] | 0.0000 |
| GPT-5.6 Luna | Authoritative reference | 94.2% | 75.3% | -18.9 | [-26.7, -11.5] | 0.0000 |
| GPT-5.6 Luna | Independent corroboration | 95.1% | 67.5% | -27.6 | [-37.4, -18.1] | 0.0000 |
| Claude Opus 5 | Unsupported user | 8.6% | 4.7% | -3.9 | [-10.1, +1.4] | 0.2490 |
| Claude Opus 5 | Individual expert | 49.0% | 15.2% | -33.7 | [-44.9, -23.5] | 0.0000 |
| Claude Opus 5 | Authoritative reference | 78.6% | 18.5% | -60.1 | [-70.8, -49.4] | 0.0000 |
| Claude Opus 5 | Independent corroboration | 65.8% | 11.5% | -54.3 | [-64.6, -44.0] | 0.0000 |
| Claude Sonnet 5 | Unsupported user | 21.8% | 68.3% | +46.5 | [+36.6, +56.4] | 0.0000 |
| Claude Sonnet 5 | Individual expert | 87.7% | 86.4% | -1.2 | [-8.6, +5.8] | 0.8279 |
| Claude Sonnet 5 | Authoritative reference | 88.9% | 91.4% | +2.5 | [-4.9, +9.9] | 0.8066 |
| Claude Sonnet 5 | Independent corroboration | 80.2% | 91.4% | +11.1 | [+3.7, +18.9] | 0.0076 |

The placement effect is not a single global shift: unsupported-user challenges rise substantially for Terra, Luna, and especially Sonnet, while expert and corroborating source effects often fall, most sharply for Opus.
Descriptively, model and source-family main effects explain 31.2% of cell-level placement-effect variance; allowing the full model-by-family interaction explains 32.8%. The remaining variation is primarily question-specific.

## Question-level transfer

| Model | Pearson r | 95% CI | Spearman rho | 95% CI |
|---|---:|---:|---:|---:|
| GPT-5.6 Terra | 0.81 | [0.72, 0.87] | 0.74 | [0.62, 0.83] |
| GPT-5.6 Luna | 0.69 | [0.58, 0.79] | 0.77 | [0.65, 0.86] |
| Claude Opus 5 | 0.49 | [0.34, 0.61] | 0.45 | [0.26, 0.61] |
| Claude Sonnet 5 | 0.58 | [0.42, 0.73] | 0.60 | [0.43, 0.73] |

## Predictive value of the matched single-turn response

In 10-fold cross-validation grouped by question, a model-and-source-family-only baseline has R2=0.357 and RMSE=0.379. Adding the matched single-turn adoption rate yields R2=0.474 and RMSE=0.343.
The within-model-family slope is 0.459 (cluster-robust SE=0.038, p=8.877e-20). Thus a 10-point increase in matched single-turn adoption predicts a 4.6-point increase in multi-turn adoption, holding model and source family fixed.

## Agreement and response distributions

- Quadratic kappa for the 0/3 to 3/3 adoption counts: 0.477.
- P(multi-turn adopts at least once | single-turn adopts at least once): 75.7%.
- P(multi-turn always adopts | single-turn always adopts): 70.1%.
- P(multi-turn adopts at least once | single-turn never adopts): 30.5%.
- Mean expected exact-answer agreement between the two three-draw empirical distributions: 72.4%.
- Mean Jensen-Shannon answer-distribution similarity: 0.753 (1 is identical).

## Important caveat

Repetition indices are exchangeable, unseeded API draws. Exact-pair agreement and McNemar results are secondary; grouped three-repetition rates are the primary estimand.
The analysis therefore does not interpret repetition 0 in one placement as a causally paired random draw with repetition 0 in the other. The primary correlations, effects, and uncertainty estimates operate on matched question-suggestion cells and cluster by question.

## Outputs

- `paired_groups.csv`: one row per matched model-question-source cell.
- `model_family_effects.csv`: paired placement effects with cluster-bootstrap intervals and FDR-adjusted tests.
- `correlations.csv`: raw, residualized, model-specific, family-specific, and model-family-specific associations.
- `question_level.csv`: question susceptibility averaged across source families.
- `analysis.json`: machine-readable audit and headline statistics.
