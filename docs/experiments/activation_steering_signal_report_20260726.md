# Prompt-only activation steering with a raw W−N mean direction

**Exploratory experiment report and next-step decision memo — 2026-07-26**

## Interpretation amendment: global estimator only

The completed experiment supports a narrower conclusion than an absence claim:
the **global, unaligned** `W−N` mean mainly captured a shared suggestion axis.
It did not test whether item-, endorsed-label-, or model-belief-conditioned
wrongness representations exist.

In particular, the small norm of the global `W−C` mean is not evidence that
item-level `W−C` information is absent. Endorsed and correct labels vary across
questions, so answer-binding components can cancel in an unaligned global
average. The follow-up
`mean_cancellation_audit_v1_20260726` therefore measures item norms,
label-conditioned banks, a sum-to-zero label-binding model, belief-conflict
directions, and low-rank structure with question-disjoint cross-validation and
within-dataset permutation placebos. This amendment supersedes any broader
reading of “no wrongness representation” below; the original numerical results
about the global vectors remain unchanged.

## Executive conclusion

This experiment does not support scaling the present `W−N` mean-direction
intervention into the original paper-sized study.

There is a reproducible probability-level effect in Llama-3.1-8B on ARC, but it
is small at the scientifically meaningful native dose, becomes visible mainly
under a very large perturbation, does not generalize to Qwen or
CommonsenseQA, and is not convincingly separated from structured
zero-expectation controls. The intervention rarely changes the top-choice
answer.

The most informative post-hoc diagnostic is geometric rather than behavioral:
at the selected layers,

\[
\cos(v_{WN},v_{CN})=0.9996
\]

for both models. The wrong-suggestion-minus-neutral and
correct-suggestion-minus-neutral directions are essentially the same vector.
The fitted vector is therefore highly likely to encode the shared suggestion
template or framing, not the correctness of the endorsed answer. The
wrong-versus-correct contrast,

\[
v_{WC}=v_{WN}-v_{CN},
\]

has only 2.5–2.7% of the `W−N` norm at the selected layers.

The mean direction itself is not noisy: item deltas are strongly aligned,
ARC- and CSQA-derived directions agree, and split-half estimates are nearly
identical. The failure is better described as **a stable estimate of the wrong
estimand** than as an unstable estimate of a useful pressure vector.

### Decision

1. Do not submit the full controlled `W−N` DAG.
2. Do not make a causal claim about a wrong-pressure-specific representation.
3. Run one bounded follow-up that contrasts wrong and correct suggestions
   directly (`W−C`) on a neutral-correct cohort, with enough matched-null seeds
   to make specificity testable.
4. Stop the global mean-direction approach if that follow-up fails at modest
   injected norm.

## Scientific question

The intended question was whether a prompt-only representation of false user
pressure causally controls agreement with the user-endorsed wrong answer while
preserving neutral knowledge.

For model layer \(\ell\), the primary direction was

\[
v_{WN,\ell}
=
\frac{1}{n}\sum_i
\left(h^W_{i,\ell}-h^N_{i,\ell}\right),
\]

and the intervention was

\[
h'_\ell=h_\ell+\alpha v_{WN,\ell}.
\]

`N` denotes the neutral prompt and `W` the ordinary wrong-suggestion prompt.
Positive \(\alpha\) was intended to amplify wrong pressure. By construction,
\(\alpha=1\) adds one raw mean prompt shift; the direction was not normalized
or scaled by a projection standard deviation.

The primary behavioral endpoint was the signed symmetric-dose effect

\[
\Delta_b(m)
=
P(b\mid W,+m)-P(b\mid W,-m),
\]

where \(b\) is the user-endorsed benchmark-wrong option. A convincing result
would require:

- positive effects in both directions relative to \(\alpha=0\);
- a useful effect at a modest injected norm;
- limited change under the neutral prompt;
- separation from norm-matched null directions;
- consistency across held-out questions, neighboring layers, datasets, and
  preferably models;
- preservation of the fixed candidate-answer probe.

## Exact experiment that was run

This was the reduced
`exploratory_benchmark_label_signal_v1_20260726` experiment, not the full
confirmatory protocol in the
[preflight specification](activation_steering_preflight.md).

The frozen configuration is
[activation_steering_signal_20260726.json](../../configs/experiments/activation_steering_signal_20260726.json),
and the cohort is
[activation_steering_signal_300_20260726.jsonl](../../configs/experiments/activation_steering_signal_300_20260726.jsonl).

### Models

- `meta-llama/Llama-3.1-8B-Instruct`
  - revision `0e9e39f249a16976918f6564b8830bc894c89659`
  - hidden size 4,096
  - nonterminal layers 1–31
- `Qwen/Qwen2.5-7B-Instruct`
  - revision `a09a35458c702b33eeacc393d103063234e8bc28`
  - hidden size 3,584
  - nonterminal layers 1–27

Each model had its own fitted vectors. No vector was transferred between
models.

### Question cohort

The deterministic cohort contained 300 questions:

| Dataset | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| ARC Challenge | 90 | 30 | 30 | 150 |
| CommonsenseQA | 90 | 30 | 30 | 150 |
| **Total** | **180** | **60** | **60** | **300** |

The stable key was `(dataset, source_example_id)`, and existing split
membership was preserved.

The endorsed option differed from the benchmark answer key. There was no
human semantic review of the endorsed wrong answer. The cohort was balanced
where possible by endorsed label and source-record neutral correctness, but
all selected CommonsenseQA rows were marked neutral-incorrect for both models
in the source metadata.

This is consequential. In the actual strict-choice evaluation at
\(\alpha=0\):

| Model | Dataset | Neutral-correct test questions |
|---|---|---:|
| Llama | ARC | 12/30 |
| Llama | CommonsenseQA | 0/30 |
| Qwen | ARC | 14/30 |
| Qwen | CommonsenseQA | 6/30 |

Thus, much of the pooled evaluation did not measure corruption of a known
answer. CommonsenseQA could not test neutral-knowledge preservation for Llama
at all.

### Prompt conditions

All strict-choice evaluations included four framings:

1. `neutral`
2. `incorrect_suggestion` (`W`)
3. `incorrect_suggestion_strong` (`S`)
4. `suggest_correct` (`C`)

The prompts ended with the frozen answer-only instruction and `Answer:`.
Tokenization then added the model-native assistant-start boundary.

### Activation and intervention site

For every framing:

1. The full user prompt, question, options, and framing were rendered with the
   model chat template.
2. No assistant answer existed yet.
3. The activation was read at the final rendered prompt token.
4. A hook added the direction to the post-decoder-block residual at that token.
5. The model scored the next-token option labels.

No assistant-answer token entered direction construction. Strict-choice
scoring used `use_cache=False`, batch size one, and did not generate text.

The pressure prompt itself necessarily contained the user's suggested answer;
the absence of answer tokens refers to the model's answer.

### Direction fitting

For each model, all four framing centroids and the following raw directions
were fit in float32 on the 180 training questions at every nonterminal layer:

\[
\begin{aligned}
v_{WN}&=\operatorname{mean}(h^W-h^N),\\
v_{CN}&=\operatorname{mean}(h^C-h^N),\\
v_{WC}&=\operatorname{mean}(h^W-h^C),\\
v_{SW}&=\operatorname{mean}(h^S-h^W).
\end{aligned}
\]

Only `W−N` was behaviorally screened and steered in this lean experiment.
The other fitted directions and all training activation states were saved,
which enabled the post-hoc geometry analysis below without additional GPU
work.

### Layer screening and selection

Every nonterminal layer was screened on validation questions:

- Llama: 31 layers × 2 datasets
- Qwen: 27 layers × 2 datasets
- total: 116 layer/dataset shards
- alphas: `[-8, -2, 0, 2, 8]`
- direction: learned native-unit `W−N` only
- no stochastic controls
- no free-generation diagnostics

Layer selection used validation data only. No test result entered selection.
Because controls and free-generation diagnostics were absent, no layer was
confirmatory-eligible; selection explicitly fell back to descriptive mode.

The selected primary layers were:

| Model | Selected layer | Held-out layers |
|---|---:|---|
| Llama | 12 | 11, 12, 13 |
| Qwen | 16 | 15, 16, 17 |

The validation result was bidirectional for Llama and not bidirectional for
Qwen. Qwen was nevertheless carried forward as the descriptive best layer.

### Held-out test matrix

Each model × dataset × test layer shard evaluated:

- 30 held-out questions;
- four prompt conditions;
- nine alphas: `[-8,-4,-2,-1,0,1,2,4,8]`;
- the learned native `W−N` vector;
- four control families with seeds 0, 1, and 2:
  - same-norm isotropic Gaussian;
  - coordinate-sign-scrambled `W−N`;
  - balanced item-sign/permutation direction in native units;
  - the same item-sign direction rescaled to `||W−N||`.

This produced

\[
30\times4\times9\times(1+4\times3)=14{,}040
\]

rows per shard and 168,480 held-out strict-choice rows across 12 shards.

### Fixed `random_all` probe

The frozen candidate-answer probe was rescored for:

- both models;
- CommonsenseQA only;
- the selected layer;
- 30 test questions;
- four prompt conditions;
- alphas `[-4,0,4]`.

Steering was applied at the saved prompt boundary. Candidate answers were then
teacher-forced, and the frozen probe was read at its original feature
layer/token. This produced 720 question/framing/dose rows.

### Aggregation and verification

- Primary report intervals used 10,000 paired question-level bootstrap
  resamples.
- The wide aggregate tables and plots used 1,000 paired bootstrap resamples.
- Control ribbons describe the three seeds; nonzero control rows were
  compacted to seed-level means and were not bootstrapped.
- Twenty recorded input/output hashes were reverified.
- All 168,480 held-out rows and 720 probe rows were present.
- All 18,960 within-shard \(\alpha=0\) comparisons had exactly zero
  probability, margin, and injected-norm change.
- No NaN or Inf occurred.
- Twenty-three focused contract/unit tests passed.

The final aggregate is stored at:

```text
/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_signal_20260726/activation_steering_signal_20260726_v1_05d7b1eae414/final_aggregate
```

The GPU artifacts were produced from commit
`c600987c23a49bf2e07b132cbf2e2988b3bd9082`. The corrected final reporting
code and plots are at commit
`6115b3a1f4bfa91884dfd7bc0733f82cfd9b7943`.

## Results

### Behavioral pressure existed before steering

The prompts themselves had a large effect at \(\alpha=0\):

| Model | Neutral `P(b)` | Wrong-suggestion `P(b)` | Natural framing gap |
|---|---:|---:|---:|
| Llama | 25.8% | 55.4% | +29.6 pp |
| Qwen | 19.2% | 36.4% | +17.1 pp |

Strong incorrect pressure produced even larger pooled `P(b)` values:

- Llama: 72.4%
- Qwen: 82.2%

The behavioral phenomenon was therefore present. The failure is not that the
models ignored user pressure.

### Primary steering result

The table uses the selected layer and pools the 30 ARC and 30 CommonsenseQA
test questions. `Neutral damage` is the paired mean absolute change in
`P(correct)` over both signs at \(|\alpha|=8\); it is not merely the difference
between group means.

| Quantity | Llama layer 12 | Qwen layer 16 |
|---|---:|---:|
| `P(b|W,−8)` | 50.70% | 33.16% |
| `P(b|W,0)` | 55.37% | 36.35% |
| `P(b|W,+8)` | 59.24% | 35.59% |
| Signed ±8 effect | **+8.55 pp** | +2.43 pp |
| 95% paired CI | **[+4.06, +13.36]** | [−0.87, +6.03] |
| Signed ±1 effect | **+1.19 pp** | +0.73 pp |
| 95% paired CI | **[+0.50, +1.94]** | [−0.19, +1.84] |
| Neutral probability damage at ±8 | 6.48 pp | 9.53 pp |
| Neutral correctness-flip rate at ±8 | 5.83% | 8.33% |

For Llama, the probability response was bidirectional:

- `P(b|W,+8)−P(b|W,0) = +3.88 pp`
- `P(b|W,0)−P(b|W,−8) = +4.67 pp`

At the native \(\alpha=1\) scale, however, `+1` moved `P(b)` only about
+0.63 pp relative to zero.

Qwen did not show the predicted pattern. `P(b|W,+8)` was below its
\(\alpha=0\) value. Its positive symmetric contrast arose because `−8`
decreased `P(b)` more, not because positive steering amplified pressure.

### The large effect required a very large intervention

At the selected layers:

| Model | `||W−N||` | Median residual norm | α=1 ratio | α=8 ratio |
|---|---:|---:|---:|---:|
| Llama | 0.643 | 6.245 | 10.3% | 82.4% |
| Qwen | 6.404 | 63.309 | 10.1% | 80.9% |

The \(\alpha=8\) dose injected a vector with roughly 81–82% of the median
residual norm. This is not a subtle intervention. At that scale, sensitivity
to generic structured perturbations is expected to become a serious
alternative explanation.

### Most of the effect did not change the chosen answer

For Llama under ordinary wrong pressure:

| Alpha | Wrong option top-1 |
|---:|---:|
| −8 | 56.7% |
| 0 | 58.3% |
| +8 | 58.3% |

The main positive result is therefore a probability or confidence shift, not
reliable control of the answer.

### Dataset and neighboring-layer consistency

At \(|\alpha|=8\):

| Model/layer | Signed effect |
|---|---:|
| Llama layer 11 | +3.08 pp |
| Llama layer 12 | +8.55 pp |
| Llama layer 13 | +8.74 pp |
| Qwen layer 15 | +0.89 pp |
| Qwen layer 16 | +2.43 pp |
| Qwen layer 17 | −6.22 pp |

Llama showed a coherent local layer neighborhood. Qwen did not.

The Llama pooled result was driven by ARC:

| Dataset | Llama signed ±8 effect | 95% CI |
|---|---:|---:|
| ARC | **+14.14 pp** | **[+7.31, +21.87]** |
| CommonsenseQA | +2.95 pp | [−2.03, +7.72] |

Neither Qwen dataset had an interval excluding zero.

### Comparison with null directions

The following values are signed ±8 effects in percentage points. Ranges are
the minimum and maximum over three seeds.

| Model | Learned W−N | Isotropic mean (range) | Coordinate-sign mean (range) | Item-sign matched mean (range) |
|---|---:|---:|---:|---:|
| Llama | **+8.55** | −1.20 (−2.16, −0.17) | +0.14 (−4.91, +4.47) | +4.26 (−0.55, **+8.38**) |
| Qwen | +2.43 | +0.90 (−3.46, +4.50) | +4.94 (+3.22, +5.84) | +3.45 (+1.82, +6.70) |

Llama clearly outperformed ordinary same-norm isotropic random vectors.
However, one structured item-sign-matched null produced +8.38 pp, nearly the
learned vector's +8.55 pp. Qwen did not outperform either the coordinate-sign
or item-sign-matched controls.

Three seeds are insufficient to estimate a 95th-percentile null. Even if all
12 heterogeneous controls were exchangeable, the smallest possible
one-sided empirical p-value would be \(1/13=0.077\). Within a three-seed
control family it would be \(1/4=0.25\). The experiment therefore could not,
by design, establish conventional direction specificity.

### Neutral-correct subset diagnostic

This analysis was performed after the primary aggregate and is descriptive.
It restricts the wrong-prompt rows to questions whose neutral \(\alpha=0\)
answer was correct.

- Llama ARC, \(n=12\):
  - signed ±1 effect: +2.92 pp;
  - signed ±8 effect: +14.76 pp;
  - wrong top-1 increased from 50.0% at zero to 58.3% at +8;
  - a matched item-sign control reached +16.42 pp.
- Llama CommonsenseQA: \(n=0\), so the intended known-answer estimand was not
  measurable.
- Qwen ARC, \(n=14\): signed ±8 effect −0.89 pp.
- Qwen CommonsenseQA, \(n=6\): signed ±8 effect −0.57 pp.

The Llama/ARC effect is not explained solely by including unknown questions,
but the subset is very small and again fails specificity against the matched
null.

### Fixed probe

The frozen probe did not degrade:

| Model | Probe top-1 at −4 | At 0 | At +4 |
|---|---:|---:|---:|
| Llama | 22.5% | 22.5% | 23.3% |
| Qwen | 34.2% | 33.3% | 35.8% |

Probe margins also did not decrease materially. This rules out obvious probe
collapse, but the low baseline top-1 rates limit the strength of any
knowledge-preservation inference.

### Numerical baseline

The critical within-shard no-op contract passed exactly. The independently
launched cross-shard replay contract did not:

- maximum option-probability difference: 0.1846;
- maximum correct-minus-endorsed margin difference: 1.625;
- top-choice agreement gate: failed.

This is consistent with BF16/GPU execution variation across independently
launched shapes or hardware. All scientific contrasts reported here use
within-shard paired rows. Cross-shard absolute comparisons are not treated as
confirmatory evidence.

## Post-hoc direction diagnosis

The saved float32 training states allow direct inspection of what the fitted
vector represents. These analyses did not use test outcomes and required no
new GPU execution.

### The mean direction is stable

At the selected layers:

| Diagnostic | Llama layer 12 | Qwen layer 16 |
|---|---:|---:|
| Median `cos(δ_i, W−N)` | 0.857 | 0.908 |
| Fraction of item deltas with positive projection | 100% | 100% |
| Median aligned energy fraction | 73.5% | 82.4% |
| Median pairwise item-delta cosine | 0.716 | 0.809 |
| Median split-half direction cosine, 1,000 splits | 0.9957 | 0.9975 |
| ARC-versus-CSQA direction cosine | 0.892 | 0.926 |

The pooled mean is highly coherent and reproducible. More training questions
would reduce estimation uncertainty, but instability of the mean is not the
main problem.

### W−N and C−N are almost identical

| Diagnostic | Llama layer 12 | Qwen layer 16 |
|---|---:|---:|
| `||W−N||` | 0.6429 | 6.4044 |
| `||C−N||` | 0.6430 | 6.4023 |
| `cos(W−N, C−N)` | **0.999644** | **0.999680** |
| `||W−C||` | 0.0172 | 0.1620 |
| `||W−C|| / ||W−N||` | **2.67%** | **2.53%** |
| `cos(W−N, W−C)` | 0.008 | 0.026 |

Both `W` and `C` use the same low-confidence suggestion template and differ
primarily in which answer is named. At the selected prompt-boundary
activations, their shared template dominates the centroid shift. Because
correct and endorsed labels vary across questions, answer-specific components
can cancel in a global unaligned mean.

Consequently, the primary vector does not isolate false pressure:

\[
v_{WN}\approx v_{CN}.
\]

It is better interpreted as a generic ordinary-suggestion or framing
direction.

This also explains why:

- the direction is geometrically stable but behaviorally weak;
- positive steering need not specifically favor the endorsed wrong answer;
- large structured null perturbations can produce comparable effects;
- Qwen can show nonmonotonic behavior despite an extremely coherent mean.

### Later layers contain a larger W−C contrast

The near-equivalence is strongest around the selected layers, but it is not
uniform across the network.

- Llama:
  - median all-layer `cos(W−N,C−N)`: 0.959;
  - maximum `||W−C||/||W−N||`: 0.566 at layer 30;
  - layers 26 and 28–31 have ratios around 0.53–0.57.
- Qwen:
  - median all-layer `cos(W−N,C−N)`: 0.9998;
  - maximum `||W−C||/||W−N||`: 0.552 at layer 25;
  - layers 21 and 23–26 have ratios around 0.53–0.55.

The original layer screen optimized the behavior of `W−N`, so it selected
layers where generic framing had leverage. It did not search for layers that
separate wrong and correct suggestions.

## Why the experiment did not work as expected

The evidence supports several distinct failure modes.

### 1. The direction formula did not isolate wrongness

This is the primary diagnosis. `W−N` changes both the presence of a suggestion
and its answer content. The shared framing change dominates. `C−N` produces
nearly the same mean vector.

### 2. The primary visible effect required an extreme dose

At \(\alpha=8\), the injected norm was approximately 81–82% of the median
residual norm. Generic perturbation sensitivity is a plausible explanation at
that scale. The native \(\alpha=1\) effect was only 1.19 pp for Llama and
non-significant for Qwen.

### 3. Specificity was underpowered

Three seeds per null family cannot estimate a tail probability. A single
matched null nearly equaled the Llama effect, and Qwen controls exceeded its
learned direction.

### 4. The cohort poorly matched the knowledge-preservation estimand

Only 32 of 120 model/question test pairs were neutral-correct. Llama had no
neutral-correct CommonsenseQA questions. The pooled result therefore mixed
known-answer corruption with changes on questions the model already answered
incorrectly.

### 5. The endpoint was mostly probabilistic

Llama's `P(b)` moved, but its wrong top-choice rate did not change in the
pooled test. This falls short if the target is behavioral answer control.

### 6. The result did not replicate

The effect was concentrated in Llama/ARC. Qwen, CommonsenseQA, and Qwen's
neighboring layers did not reproduce it.

### 7. The fixed probe guardrail was weak

The probe did not deteriorate, but its low baseline top-1 performance makes
invariance difficult to interpret as preservation of actionable knowledge.

## Recommended next checks

The next work should be a decisive, bounded diagnosis—not a larger repetition
of `W−N`.

### Priority 0: finish the no-GPU representation audit

Use the already-saved training states to produce layerwise, modelwise versions
of:

- `cos(W−N,C−N)`;
- `||W−C||/||W−N||`;
- split-half and bootstrap stability for `W−C`;
- item-level `cos(h_i^W-h_i^C, v_{WC})`;
- ARC-versus-CSQA `W−C` direction cosine;
- centered identity-versus-framing ratios;
- answer-label-conditional means to test whether global label balancing
  cancels the relevant signal.

Decision rule:

- If `W−C` is unstable across splits or datasets, do not test a global
  one-dimensional correctness direction. Move to an item-conditioned or
  low-rank subspace formulation.
- If `W−C` is stable at late layers, proceed to the bounded behavioral gate
  below.

### Priority 1: run a focused W−C behavioral gate

Recommended initial scope:

- model: Llama only;
- dataset: ARC only;
- evaluation: a new question-disjoint set of at least 100
  neutral-\(\alpha=0\)-correct questions;
- layers: a training-geometry-selected late-layer set, initially
  26, 28, 29, 30, and 31;
- conditions: neutral, wrong suggestion, and correct suggestion;
- learned directions:
  - native `W−C`;
  - `W−C` rescaled to the `W−N` norm, labeled separately;
  - `W−N` and `C−N` as diagnostic comparators;
- doses chosen so the injected/residual ratio does not exceed 0.20 for the
  primary analysis;
- strict choice scoring, batch size one, same-shard paired baselines.

The cohort may be fixed using neutral correctness because it is measured
before steering. It must not be selected using any steered response. For a
confirmatory semantic claim, the endorsed wrong answers should also be human
reviewed.

### Priority 2: make the null comparison statistically meaningful

For the selected `W−C` layer and dose, generate matched controls from the
item-level `W−C` deltas, not from `W−N` deltas.

- Minimum: 20 randomization seeds per null family. With 20 exchangeable nulls,
  the smallest one-sided empirical p-value is \(1/21\approx0.0476\).
- Preferred: 50 seeds for a more stable tail estimate.
- Primary null: balanced item-sign/permutation, rescaled to
  `||W−C||`.
- Secondary null: same-norm isotropic vectors.

This is not a return to the original enormous matrix. Restricting to one
model, one dataset, a few layers, three conditions, and modest doses makes
20–50 seeds substantially cheaper than the completed all-layer experiment.

### Priority 3: predeclare a stop/go criterion

Proceed beyond the focused gate only if all of the following hold on new
held-out questions:

1. The native or modest-norm `W−C` direction is bidirectional relative to
   \(\alpha=0\).
2. Its paired 95% interval excludes zero.
3. It exceeds the predeclared 95th percentile of the matched-null
   randomization distribution.
4. The result changes wrong-answer top-1 behavior, not only probabilities.
5. Mean absolute neutral `P(correct)` damage is smaller than the pressure
   effect, and the neutral correctness-flip rate is at most 2%.
6. The primary injected/residual norm ratio is at most 0.20.
7. The effect is present at the selected layer and at least one neighbor.

If these criteria fail, stop global mean-vector steering.

### Priority 4: only after W−C passes, test mechanism variants

If a specific `W−C` effect exists, then test:

- `S−W` to isolate pressure strength while holding the wrong answer fixed;
- residualized `W−N` after projecting out `C−N`;
- multiple final prompt tokens versus the single boundary token;
- free-generation answer validity and repetition;
- replication in Qwen and a repaired CommonsenseQA cohort;
- fixed-probe behavior on a probe with adequate baseline accuracy.

These should not precede the specificity gate. Otherwise they would multiply
analyses around a direction already shown to be dominated by generic framing.

## Minimal next-experiment matrix

A practical two-stage design is:

### Stage A: late-layer validation screen

- Llama, ARC
- at least 100 neutral-correct validation questions
- five late layers
- three conditions
- five modest doses
- learned `W−C`, `W−N`, and `C−N`

Approximate strict-choice rows:

\[
100\times5\times3\times5\times3=22{,}500.
\]

### Stage B: selected-layer specificity test

- one selected layer plus one neighbor;
- 100 new held-out neutral-correct questions;
- three conditions;
- five doses;
- learned `W−C`;
- 20 item-sign-matched controls and 20 isotropic controls.

Approximate strict-choice rows:

\[
100\times2\times3\times5\times(1+40)=123{,}000.
\]

This is comparable to, or smaller than, the completed lean test and directly
answers the unresolved question. It should be run only if the CPU-only
`W−C` stability audit is favorable.

## Final assessment

The completed experiment did identify a stable prompt-framing displacement and
showed that perturbing it can move Llama answer probabilities. It did not
identify a robust, general, wrong-pressure-specific causal direction.

The most plausible explanation is not merely inadequate sample size:

- the direction estimate is exceptionally stable;
- `W−N` and `C−N` are almost identical;
- the correctness-specific residual is tiny at the selected layers;
- the large behavioral effect requires an intervention near the scale of the
  residual stream;
- structured zero-expectation controls can match it.

The next scientific question should therefore be:

> Is there a stable, behaviorally specific `W−C` or residualized
> correctness/endorsement direction at later layers on questions the model
> demonstrably answers correctly without pressure?

That question can be answered with one focused follow-up. A negative answer
would justify abandoning one-dimensional global mean steering for this
mechanism and moving to item-conditioned or low-rank representations.
