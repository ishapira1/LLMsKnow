# Strict MC Definitions

This note documents the active definitions used by the codebase for `strict_mc` multiple-choice evaluation.

It is only about the `strict_mc` path. It does not describe the older free-form multiple-choice grading path.

## Core idea

In `strict_mc`, we do not sample a free-form answer from the model.

Instead, we:

1. look at the model's next-token probabilities over the allowed answer letters
2. renormalize probability mass over those allowed letters only
3. choose the highest-probability letter as the model's selected choice

So the main hard evaluation for `strict_mc` is based on the argmax choice, not on stochastic sampling.

## Definitions

### Allowed choices

The allowed choices are the answer letters permitted for the prompt, usually from the row's `letters` field, such as `A`, `B`, `C`, `D`.

### `choice_probabilities`

`choice_probabilities` is the model's probability distribution over the allowed answer letters only, after renormalizing within that set so the probabilities sum to 1.

Example:

- `P(A) = 0.18`
- `P(B) = 0.34`
- `P(C) = 0.29`
- `P(D) = 0.19`

These four numbers sum to 1 over the allowed choices.

### `selected choice`

The `selected choice` is the allowed answer letter with the highest probability in `choice_probabilities`.

This is:

- a deterministic argmax
- not a sampled draw
- allowed to be below 50%

So if the top option has probability `0.34`, that option is still the selected choice.

If there is an exact tie, the earlier choice in the option order wins.

### `response_raw`

For `strict_mc` choice-scoring rows, `response_raw` is the selected letter itself.

This is different from generation-based paths, where `response_raw` is a full text completion.

### `correct_letter`

`correct_letter` is the gold answer letter from the dataset metadata.

### `P(correct)`

`P(correct)` is the probability assigned to the gold answer letter.

This is a soft metric. It tells us how much probability mass the model assigns to the correct option, even when that option is not the top choice.

### `P(selected)`

`P(selected)` is the probability assigned to the selected choice, meaning the top-probability answer letter.

### `correctness`

`correctness` is the hard row-level label used for grading.

- `1` if `selected choice == correct_letter`
- `0` if `selected choice != correct_letter`
- `null` only if the row is unusable or ambiguous

For the normal `strict_mc` choice-scoring path, this is usually just the correctness of the argmax choice.

### `accuracy`

`accuracy` is the mean of `correctness` across rows.

So in `strict_mc`, accuracy means:

"How often does the top-probability answer letter equal the gold answer letter?"

This is not the same as averaging `P(correct)`.

### `usable_for_metrics`

`usable_for_metrics` indicates whether the row is allowed into:

- accuracy summaries
- paired tuple construction
- probe training and evaluation

Ambiguous or unusable rows are excluded from those downstream metrics.

### `n_draws`

For `strict_mc` choice-scoring, `n_draws` is effectively forced to `1`.

That is because the pipeline is not sampling repeated completions in this mode. It computes one deterministic choice distribution and one selected choice per prompt.

### `T_prompt`

`T_prompt` is a prompt-level success field.

For `strict_mc`, the definition is:

- `T_prompt = P(correct)`

So in `strict_mc`, `T_prompt` is the model's probability mass on the gold answer choice.

This can be confusing because in generation-based paths, `T_prompt` means something different: empirical accuracy across repeated draws.

For this reason, when reading `strict_mc` outputs, it is safest to think of:

- `accuracy` as the hard argmax metric
- `T_prompt` as the soft gold-choice probability

## Worked example

Suppose the allowed choices are `A/B/C/D`, and the model's strict-MC choice probabilities are:

- `P(A) = 0.18`
- `P(B) = 0.34`
- `P(C) = 0.29`
- `P(D) = 0.19`

Suppose the gold answer is:

- `correct_letter = C`

Then:

- `selected choice = B`
- `P(correct) = 0.29`
- `P(selected) = 0.34`
- `correctness = 0`
- this row contributes `0` to `accuracy`
- `T_prompt = 0.29`

If instead the gold answer were `B`, then:

- `selected choice = B`
- `P(correct) = 0.34`
- `P(selected) = 0.34`
- `correctness = 1`
- this row contributes `1` to `accuracy`
- `T_prompt = 0.34`

## Short summary

For `strict_mc`:

- `selected choice` = highest-probability allowed letter
- `P(correct)` = probability mass on the gold letter
- `accuracy` = mean argmax correctness
- `T_prompt` = same as `P(correct)` in this mode
