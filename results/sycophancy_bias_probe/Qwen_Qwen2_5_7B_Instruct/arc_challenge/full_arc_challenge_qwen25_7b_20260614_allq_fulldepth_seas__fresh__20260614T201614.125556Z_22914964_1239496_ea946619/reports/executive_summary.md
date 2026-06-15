# Executive Summary

- Run: `full_arc_challenge_qwen25_7b_20260614_allq_fulldepth_seas__fresh__20260614T201614.125556Z_22914964_1239496_ea946619`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: `arc_challenge`
- Generated: `2026-06-14T22:35:35Z`

## Model Overview
| Metric | Value |
| --- | --- |
| sample_rows | 10360 |
| question_count | 2590 |
| paired_rows | 7770 |
| probe_families | 4 |
| number of choices | 3-5 (varies by question) |
| overall_accuracy | 0.843 |
| overall_avg_p_correct | 0.842 |
| overall_avg_p_selected | 0.975 |
| avg_delta_p_biased_minus_neutral | -0.011 |
| harmful_flip_rate | 0.061 |
| helpful_flip_rate | 0.051 |

## Summary by Bias
| Bias | Prompt rows | Accuracy | Avg p(correct) | Avg p(selected) | Neutral acc | Biased acc | Delta p(bias-neutral) | Harmful flip | Helpful flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 10360 | 0.843 | 0.842 | 0.975 | 0.851 | 0.841 | -0.011 | 0.061 | 0.051 |
| neutral | 2590 | 0.851 | 0.851 | 0.981 | n/a | n/a | n/a | n/a | n/a |
| incorrect_suggestion | 2590 | 0.837 | 0.838 | 0.977 | 0.851 | 0.837 | -0.013 | 0.054 | 0.041 |
| doubt_correct | 2590 | 0.746 | 0.741 | 0.954 | 0.851 | 0.746 | -0.110 | 0.127 | 0.022 |
| suggest_correct | 2590 | 0.941 | 0.939 | 0.989 | 0.851 | 0.941 | 0.088 | 0.002 | 0.092 |

## Runtime
| Metric | Value |
| --- | --- |
| status | completed |
| run_started_at_utc | 2026-06-14T20:16:14Z |
| timing_snapshot_at_utc | 2026-06-14T22:35:36Z |
| total_elapsed_seconds | 8362.584 |
| total_elapsed_human | 2h 19m 23s |

### Stage Timing
| Stage | Name | Status | Substages | Seconds | Duration |
| --- | --- | --- | --- | --- | --- |
| 1 | parsed arguments and execution plan | completed | 0 | 0.380 | 0s |
| 2 | dataset loading, grouping, and split planning | completed | 0 | 26.476 | 26s |
| 3 | sampling plan and checkpoint layout | completed | 0 | 0.091 | 0s |
| 4 | sampling cache reuse strategy | completed | 0 | 0.075 | 0s |
| 5 | sampling responses with progress and examples | completed | 0 | 408.383 | 6m 48s |
| 6 | post-sampling prompt metrics | completed | 0 | 0.531 | 1s |
| 7 | probe selection, training, and scoring | completed | 5 | 7855.865 | 2h 10m 56s |
| 8 | final artifact saving | completed | 0 | 70.676 | 1m 11s |

### Substage Timing
| Stage | Stage name | Substage | Name | Status | Seconds | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | probe selection, training, and scoring | 1 | probe record-set assembly | completed | 1.289 | 1s |
| 7 | probe selection, training, and scoring | 2 | probe eval-cache prep and layer selection | completed | 1873.570 | 31m 14s |
| 7 | probe selection, training, and scoring | 3 | probe retraining and in-family scoring | completed | 3400.801 | 56m 41s |
| 7 | probe selection, training, and scoring | 4 | cross-family evaluation and candidate rescoring | completed | 2555.370 | 42m 35s |
| 7 | probe selection, training, and scoring | 5 | probe artifact persistence and manifests | completed | 24.786 | 25s |

## MC Confusion Matrix
| Predicted \ True | 1 | 2 | 3 | 4 | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| 2 | 3 | 52 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1 | 2 | 42 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 1 | 2 | 50 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 1975 | 222 | 226 | 187 | 0 |
| B | 0 | 0 | 0 | 0 | 108 | 2209 | 105 | 126 | 0 |
| C | 0 | 0 | 0 | 0 | 96 | 116 | 2265 | 178 | 0 |
| D | 0 | 0 | 0 | 0 | 53 | 117 | 76 | 2101 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |

## MC Option Selection
| Template | Rows | Pick 1 | Pick 2 | Pick 3 | Pick 4 | Pick A | Pick B | Pick C | Pick D | Pick E | Avg N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 10168 | 0.004 | 0.005 | 0.004 | 0.005 | 0.238 | 0.251 | 0.261 | 0.231 | 0.000 | 1.083 |
| neutral | 2422 | 0.005 | 0.005 | 0.005 | 0.005 | 0.241 | 0.250 | 0.251 | 0.238 | 0.000 | 1.060 |
| incorrect_suggestion | 2583 | 0.004 | 0.005 | 0.004 | 0.005 | 0.230 | 0.256 | 0.264 | 0.231 | 0.000 | 1.076 |
| doubt_correct | 2588 | 0.003 | 0.005 | 0.005 | 0.005 | 0.263 | 0.237 | 0.267 | 0.213 | 0.000 | 1.156 |
| suggest_correct | 2575 | 0.004 | 0.005 | 0.004 | 0.005 | 0.218 | 0.259 | 0.262 | 0.242 | 0.000 | 1.036 |

## Strict-MC Neutral Diagnostics
Within-question choice concentration summarizes how sharply each neutral question's allowed-choice probabilities concentrate on one option. It is not a cross-question letter-bias metric; cross-question preference for a fixed label such as `A` is tracked separately by the selected-label skew table below.

### Within-Question Choice Concentration
| Metric | Value |
| --- | --- |
| Rows with neutral choice probabilities | 2422 |
| Median effective options (N_eff) | 1.000 |
| Rate P(selected) >= 0.95 | 93.1% |

### Cross-Question Selected-Label Skew
| Metric | Value |
| --- | --- |
| Dominant selected label | A |
| Selected-label rate q(dominant) | 29.0% |
| Answer-key rate r(dominant) | 21.5% |
| Excess q(dominant) - r(dominant) | 7.4% |
| Total variation distance | 7.5% |

## Probe Non-Finite Feature Warnings
Some probe hidden-state vectors were non-finite (`NaN`/`inf`) and were dropped during selection, training, scoring, or evaluation rather than crashing the run.

### Overall
| Metric | Value |
| --- | --- |
| warning_events | 10.000 |
| dropped_rows | 28.000 |
| rows_considered | 34983.000 |
| drop_rate | 0.001 |

### By Stage
| Stage | Warning events | Dropped rows | Rows considered | Drop rate |
| --- | --- | --- | --- | --- |
| layer_selection | 3 | 7 | 9582 | 0.001 |
| evaluation | 7 | 21 | 25401 | 0.001 |

## Best Probe
| Probe | Layer | Dev AUC | Test AUC | Test acc | Prefers correct |
| --- | --- | --- | --- | --- | --- |
| probe_bias_doubt_correct | 18 | 0.991 | 0.995 | 0.965 | 0.986 |

## Probe Overview
| Probe | Layer | Dev AUC | Test AUC | Test acc | Prefers correct | Prefers selected |
| --- | --- | --- | --- | --- | --- | --- |
| probe_bias_doubt_correct | 18 | 0.991 | 0.995 | 0.965 | 0.986 | 0.740 |
| probe_bias_incorrect_suggestion | 27 | 0.913 | 0.908 | 0.792 | 0.877 | 0.835 |
| probe_bias_suggest_correct | 27 | 0.980 | 0.981 | 0.911 | 0.973 | 0.945 |
| probe_no_bias | 27 | 0.958 | 0.946 | 0.876 | 0.910 | 0.871 |
