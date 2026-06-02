# Claim 3 first-pass summary

## Neutral-trained probe pooled metrics
| test_framing               |   n_questions |   top1_acc |   K_pairwise |   gold_vs_bestwrong_margin_mean |
|:---------------------------|--------------:|-----------:|-------------:|--------------------------------:|
| doubt_correct              |         27104 |   0.616957 |     0.832291 |                        0.161532 |
| incorrect_suggestion       |         27104 |   0.656139 |     0.853503 |                        0.233152 |
| model_congruent_suggestion |          1172 |   0.882253 |     0.938567 |                        0.598803 |
| neutral                    |         27104 |   0.810729 |     0.923935 |                        0.476299 |
| suggest_correct            |         27104 |   0.930158 |     0.976403 |                        0.635287 |

## Model pooled metrics
| test_framing               |   n_questions |   top1_acc |   K_pairwise |   gold_vs_bestwrong_margin_mean |   gold_vs_endorsed_margin_mean |
|:---------------------------|--------------:|-----------:|-------------:|--------------------------------:|-------------------------------:|
| doubt_correct              |         27051 |   0.522716 |     0.78336  |                       0.0674157 |                    nan         |
| incorrect_suggestion       |         26858 |   0.651836 |     0.855167 |                       0.313614  |                      0.500372  |
| model_congruent_suggestion |          1163 |   0.8908   |     0.935512 |                       0.782279  |                     -0.0585309 |
| neutral                    |         25669 |   0.776735 |     0.904837 |                       0.552577  |                    nan         |
| suggest_correct            |         26761 |   0.933224 |     0.974789 |                       0.849067  |                      0         |

## Neutral-trained probe same-candidate shifts: incorrect_suggestion
| choice_type    |     n |   mean_delta |   mean_abs_delta |
|:---------------|------:|-------------:|-----------------:|
| correct        | 27104 |   -0.161425  |         0.191017 |
| endorsed_wrong | 27090 |    0.150798  |         0.170554 |
| other_wrong    | 76140 |   -0.0201829 |         0.074672 |

## Neutral-trained probe same-candidate shifts: suggest_correct
| choice_type   |      n |   mean_delta |   mean_abs_delta |
|:--------------|-------:|-------------:|-----------------:|
| correct       |  27104 |    0.0756127 |        0.110008  |
| other_wrong   | 103230 |   -0.0320324 |        0.0824739 |

## Neutral-trained probe same-candidate shifts: doubt_correct
| choice_type   |      n |   mean_delta |   mean_abs_delta |
|:--------------|-------:|-------------:|-----------------:|
| correct       |  27104 |   -0.261561  |        0.282094  |
| wrong         | 103230 |    0.0210851 |        0.0732178 |

## Neutral-trained probe score stability
| target_framing             |      n |   pearson_all |   pearson_correct |   pearson_wrong |   spearman_all |
|:---------------------------|-------:|--------------:|------------------:|----------------:|---------------:|
| doubt_correct              | 130334 |      0.760341 |          0.636631 |        0.730353 |       0.858778 |
| incorrect_suggestion       | 130334 |      0.781023 |          0.739632 |        0.602207 |       0.820965 |
| model_congruent_suggestion |   4687 |      0.887688 |          0.808443 |        0.716078 |       0.924001 |
| suggest_correct            | 130334 |      0.872092 |          0.768134 |        0.672492 |       0.867248 |