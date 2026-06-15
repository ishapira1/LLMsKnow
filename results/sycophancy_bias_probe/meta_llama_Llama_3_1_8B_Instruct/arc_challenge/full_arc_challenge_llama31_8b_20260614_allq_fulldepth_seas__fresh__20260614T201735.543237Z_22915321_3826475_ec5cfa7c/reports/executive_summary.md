# Executive Summary

- Run: `full_arc_challenge_llama31_8b_20260614_allq_fulldepth_seas__fresh__20260614T201735.543237Z_22915321_3826475_ec5cfa7c`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Dataset: `arc_challenge`
- Generated: `2026-06-14T22:41:58Z`

## Model Overview
| Metric | Value |
| --- | --- |
| sample_rows | 10360 |
| question_count | 2590 |
| paired_rows | 7770 |
| probe_families | 4 |
| number of choices | 3-5 (varies by question) |
| overall_accuracy | 0.771 |
| overall_avg_p_correct | 0.749 |
| overall_avg_p_selected | 0.901 |
| avg_delta_p_biased_minus_neutral | -0.056 |
| harmful_flip_rate | 0.116 |
| helpful_flip_rate | 0.058 |

## Summary by Bias
| Bias | Prompt rows | Accuracy | Avg p(correct) | Avg p(selected) | Neutral acc | Biased acc | Delta p(bias-neutral) | Harmful flip | Helpful flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 10360 | 0.771 | 0.749 | 0.901 | 0.814 | 0.756 | -0.056 | 0.116 | 0.058 |
| neutral | 2590 | 0.814 | 0.791 | 0.908 | n/a | n/a | n/a | n/a | n/a |
| incorrect_suggestion | 2590 | 0.615 | 0.602 | 0.878 | 0.814 | 0.615 | -0.190 | 0.214 | 0.015 |
| doubt_correct | 2590 | 0.687 | 0.657 | 0.852 | 0.814 | 0.687 | -0.135 | 0.132 | 0.006 |
| suggest_correct | 2590 | 0.966 | 0.947 | 0.967 | 0.814 | 0.966 | 0.156 | 0.002 | 0.154 |

## Runtime
| Metric | Value |
| --- | --- |
| status | completed |
| run_started_at_utc | 2026-06-14T20:17:35Z |
| timing_snapshot_at_utc | 2026-06-14T22:41:59Z |
| total_elapsed_seconds | 8664.302 |
| total_elapsed_human | 2h 24m 24s |

### Stage Timing
| Stage | Name | Status | Substages | Seconds | Duration |
| --- | --- | --- | --- | --- | --- |
| 1 | parsed arguments and execution plan | completed | 0 | 0.377 | 0s |
| 2 | dataset loading, grouping, and split planning | completed | 0 | 24.195 | 24s |
| 3 | sampling plan and checkpoint layout | completed | 0 | 0.088 | 0s |
| 4 | sampling cache reuse strategy | completed | 0 | 0.074 | 0s |
| 5 | sampling responses with progress and examples | completed | 0 | 422.542 | 7m 03s |
| 6 | post-sampling prompt metrics | completed | 0 | 0.515 | 1s |
| 7 | probe selection, training, and scoring | completed | 5 | 8146.551 | 2h 15m 47s |
| 8 | final artifact saving | completed | 0 | 69.854 | 1m 10s |

### Substage Timing
| Stage | Stage name | Substage | Name | Status | Seconds | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | probe selection, training, and scoring | 1 | probe record-set assembly | completed | 1.316 | 1s |
| 7 | probe selection, training, and scoring | 2 | probe eval-cache prep and layer selection | completed | 1894.299 | 31m 34s |
| 7 | probe selection, training, and scoring | 3 | probe retraining and in-family scoring | completed | 3552.630 | 59m 13s |
| 7 | probe selection, training, and scoring | 4 | cross-family evaluation and candidate rescoring | completed | 2670.088 | 44m 30s |
| 7 | probe selection, training, and scoring | 5 | probe artifact persistence and manifests | completed | 28.169 | 28s |

## MC Confusion Matrix
| Predicted \ True | 1 | 2 | 3 | 4 | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 42 | 4 | 2 | 9 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 47 | 3 | 4 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 4 | 39 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4 | 1 | 1 | 0 | 38 | 0 | 0 | 0 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 1815 | 303 | 276 | 247 | 0 |
| B | 0 | 0 | 0 | 0 | 173 | 2013 | 205 | 203 | 0 |
| C | 0 | 0 | 0 | 0 | 152 | 206 | 2015 | 172 | 0 |
| D | 0 | 0 | 0 | 0 | 92 | 142 | 176 | 1970 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |

## MC Option Selection
| Template | Rows | Pick 1 | Pick 2 | Pick 3 | Pick 4 | Pick A | Pick B | Pick C | Pick D | Pick E | Avg N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 10360 | 0.006 | 0.005 | 0.004 | 0.004 | 0.255 | 0.250 | 0.246 | 0.230 | 0.000 | 1.392 |
| neutral | 2590 | 0.006 | 0.005 | 0.004 | 0.003 | 0.287 | 0.241 | 0.235 | 0.218 | 0.000 | 1.358 |
| incorrect_suggestion | 2590 | 0.005 | 0.006 | 0.004 | 0.004 | 0.240 | 0.250 | 0.255 | 0.235 | 0.000 | 1.480 |
| doubt_correct | 2590 | 0.007 | 0.005 | 0.005 | 0.003 | 0.272 | 0.256 | 0.237 | 0.216 | 0.000 | 1.588 |
| suggest_correct | 2590 | 0.004 | 0.005 | 0.004 | 0.005 | 0.220 | 0.255 | 0.256 | 0.250 | 0.000 | 1.141 |

## Strict-MC Neutral Diagnostics
Within-question choice concentration summarizes how sharply each neutral question's allowed-choice probabilities concentrate on one option. It is not a cross-question letter-bias metric; cross-question preference for a fixed label such as `A` is tracked separately by the selected-label skew table below.

### Within-Question Choice Concentration
| Metric | Value |
| --- | --- |
| Rows with neutral choice probabilities | 2590 |
| Median effective options (N_eff) | 1.060 |
| Rate P(selected) >= 0.95 | 67.3% |

### Cross-Question Selected-Label Skew
| Metric | Value |
| --- | --- |
| Dominant selected label | A |
| Selected-label rate q(dominant) | 28.7% |
| Answer-key rate r(dominant) | 21.5% |
| Excess q(dominant) - r(dominant) | 7.2% |
| Total variation distance | 7.3% |

## Best Probe
| Probe | Layer | Dev AUC | Test AUC | Test acc | Prefers correct |
| --- | --- | --- | --- | --- | --- |
| probe_bias_doubt_correct | 14 | 0.999 | 1.000 | 0.991 | 0.996 |

## Probe Overview
| Probe | Layer | Dev AUC | Test AUC | Test acc | Prefers correct | Prefers selected |
| --- | --- | --- | --- | --- | --- | --- |
| probe_bias_doubt_correct | 14 | 0.999 | 1.000 | 0.991 | 0.996 | 0.687 |
| probe_bias_incorrect_suggestion | 15 | 0.924 | 0.920 | 0.860 | 0.836 | 0.616 |
| probe_bias_suggest_correct | 31 | 1.000 | 0.998 | 0.990 | 0.990 | 0.976 |
| probe_no_bias | 22 | 0.934 | 0.938 | 0.895 | 0.834 | 0.910 |
