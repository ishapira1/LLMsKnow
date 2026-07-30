# Simultaneous Multilayer Conditioned-Steering Gate

Date: 2026-07-29

## Question

Does the saved ARC belief-conflict direction become causally effective when its
layer-specific vector is added at every nonterminal residual layer in the same
forward pass, rather than at one layer at a time?

## Design

- Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.
- Data: the frozen model-specific, source-neutral-correct ARC validation
  cohorts from the conditioned-steering experiment.
- Cohort after same-shard BF16 replay: 120 Llama questions and 119 Qwen
  questions.
- Conditions: neutral (`N`), ordinary wrong suggestion (`W`), and ordinary
  correct suggestion (`C`).
- Direction: the saved ARC belief-conflict direction, separately estimated at
  each residual layer from the existing training activations.
- All-layer site: residual layers 1–31 for Llama and 1–27 for Qwen.
- Same-shard comparator: the previously selected single layer, L16 for Llama
  and L21 for Qwen.
- Prompt positions: final assistant-start boundary only, and the
  energy-matched prompt suffix.
- Aggregate normalized ratio:
  `[-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]`.
- All-layer scaling: each layer received
  `aggregate_ratio / sqrt(number_of_layers)`, measured relative to that
  layer's median ARC-training neutral residual norm. The root-sum-square of
  normalized layer doses therefore equals the declared aggregate ratio.
- Strict-choice scoring used batch size one and `use_cache=False`.
- Primary negative dose was preregistered as the correction sign. Eligibility
  required bidirectionality, at least a 5-point fall in wrong-prompt top-1
  endorsement, at most 2 points of neutral and correct-suggestion accuracy
  damage, and a paired difference-in-differences interval above zero.

Implementation commit: `87665f5c161a0d5c34d3514e2a506cbae072313b`.

## Numerical gates and execution

- BF16/no-op Slurm array: `36195800`; both tasks completed.
- Validation Slurm array: `36196337`; both tasks completed.
- CPU selection job: `36197164`; completed.
- Alpha zero was an exact same-shard no-op for every all-layer token mask in
  both the BF16 gate and full validation.
- No nonfinite values occurred.
- Llama produced 10,080 question-level rows from 9,720 forwards.
- Qwen produced 9,996 question-level rows from 9,640 forwards.
- Saved result hashes match their manifests.

## Natural alpha-zero behavior

| Model | Condition | Mean P(b) | Mean P(c) | Top-1 b | Accuracy |
|---|---:|---:|---:|---:|---:|
| Llama | N | 0.0188 | 0.9277 | 0.0% | 100.0% |
| Llama | W | 0.2824 | 0.6598 | 29.2% | 67.5% |
| Llama | C | 0.0045 | 0.9790 | 0.0% | 100.0% |
| Qwen | N | 0.0022 | 0.9834 | 0.0% | 100.0% |
| Qwen | W | 0.0579 | 0.9141 | 5.9% | 91.6% |
| Qwen | C | 0.0001 | 0.9997 | 0.0% | 100.0% |

The validation cohorts therefore contain a strong natural wrong-suggestion
effect for Llama and a smaller one for Qwen.

## Preregistered result

Neither model had an eligible candidate. `both_models_pass=false`.

The negative direction did not consistently reduce `P(b|W)`, positive steering
did not consistently increase it, and no treatment reduced wrong-prompt top-1
endorsement by the required 5 points.

At aggregate ratio 0.20:

- Llama all-layer boundary negative steering increased `P(b|W)` by 0.67
  points, increased top-1 endorsement by 0.83 points, and reduced neutral
  accuracy by 30 points.
- Qwen all-layer boundary negative steering increased `P(b|W)` by 1.16
  points, increased top-1 endorsement by 0.84 points, and reduced neutral
  accuracy by 3.36 points.
- Energy-matched suffix intervention caused less neutral damage, but its
  behavioral effects remained small and/or in the wrong direction.

The positive difference-in-differences observed for Llama all-layer boundary at
negative ratio 0.20 (`0.0812`, paired 95% interval `[0.0223, 0.1407]`) is not a
successful correction: it is driven by much larger changes on neutral prompts
and accompanies 30-point neutral accuracy damage.

## Exploratory sign-flipped diagnostic

The observed causal sign was often opposite to the estimator's intended sign,
so positive doses were also examined post hoc as possible correction doses.
This is diagnostic and was not used to declare a preregistered pass.

- Llama all-layer boundary at `+0.20` reduced mean `P(b|W)` by 5.08 points,
  but reduced top-1 endorsement by only 0.83 points, reduced neutral accuracy
  by 15.83 points, reduced correct-suggestion accuracy by 8.33 points, and
  changed neutral `P(c)` by 29.40 points on average.
- At Llama `+0.05`, the wrong-prompt probability reduction was 1.32 points
  and top-1 reduction was 1.67 points, already with 2.5 points of neutral
  accuracy damage and 4.44 points of mean absolute neutral `P(c)` damage.
- Qwen all-layer energy-matched suffix at `+0.20` reduced mean `P(b|W)` by
  0.68 points and top-1 endorsement by 0.84 points, with 0.84 points of neutral
  accuracy damage. Its sign-flipped difference-in-differences was `0.00683`
  with paired 95% interval `[0.00134, 0.01359]`.

Thus the all-layer intervention can move probabilities, especially for Llama,
but the Llama movement is nonspecific and damaging, while the low-damage Qwen
movement is too small to provide meaningful top-choice control.

## Conclusion

Simultaneous steering at every nonterminal layer does not rescue the
one-vector conditioned correction in this setup.

The experiment rules out single-position depth as the main explanation for the
previous null result: repeating layer-specific belief-conflict additions across
depth produces either broad disruption (Llama boundary steering) or a small
probability-only effect (Qwen suffix steering), not a robust reduction in
sycophantic answer choices. No held-out or control DAG is authorized from this
validation.

This result does not show that answer- or belief-specific information is absent.
It shows that repeatedly adding the current mean conditioned direction is not a
selective behavioral control. A genuinely different intervention family—such
as item-aligned/answer-binding subspaces, causal patching, or learned
low-rank/erasure operators—would be needed to test that remaining hypothesis.

## Artifacts

Harvard result root:

```text
/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/
sycophancy_bias_intervention/activation_steering_multilayer_gate_20260729/
activation_steering_multilayer_gate_20260729_v1_8986aa239e34
```

Key files:

- `models/llama31_8b/validation/question_results.jsonl`
- `models/llama31_8b/validation/manifest.json`
- `models/qwen25_7b/validation/question_results.jsonl`
- `models/qwen25_7b/validation/manifest.json`
- `validation_decision.json`
