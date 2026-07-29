# Addressee-indexed endorsement experiment

This experiment tests whether GPT-5.4-nano follows an incorrect endorsement
more strongly when the person expressing the belief is also the person who
will receive the response.

## Frozen protocol

- Model snapshot: `gpt-5.4-nano-2026-03-17`
- Datasets: CommonsenseQA and ARC Challenge
- Cohort: 250 neutral-correct questions from each dataset
- Seed: `5`
- API: Chat Completions with `reasoning_effort="none"`,
  `temperature=1`, `top_p=1`, `logprobs=true`, `top_logprobs=5`, and
  `max_completion_tokens=32`
- Historical reuse: CommonsenseQA neutral and regular incorrect-suggestion
  responses from
  `anti_sycophancy_request_gpt54nano_incorrect_live_20260624`
- Uncertainty: 10,000 question-paired bootstrap resamples

The four factorial prompts mention both the user and the customer and inject
the full text of the designated incorrect option after the question/options
and before the answer-only instruction.

## Reproduction

```bash
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py prepare
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py count
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py estimate
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py run-live \
  --confirm-spend --max-cost-usd 2.0
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py analyze
PYTHONPATH=src python scripts/run_openai_addressee_indexing.py audit
```

Paid execution is rejected without `--confirm-spend`. The user-provided cost
cap must be strictly below `$10`, and the conservative estimate must be below
both that cap and the absolute `$10` ceiling.

## Results

The experiment completed with 500 questions and 3,000 condition-level
observations. The actual new API cost was `$0.0934`; the deliberately
conservative pre-run upper bound was `$1.4417`.

The primary equal-weight matched-minus-mismatched contrast was `+0.6`
percentage points in sycophancy drop, with a 95% paired-bootstrap interval of
`[-0.5, +1.7]`. The endorsement effect therefore was not meaningfully indexed
to the addressee in this experiment.

The identity of the believer mattered substantially more. Sycophancy was
`13.4` percentage points higher when the user rather than the customer was
described as holding the belief, with a 95% interval of `[10.6, 16.4]`. This
supports recipient invariance for these prompts, while also identifying a
believer-identity or wording effect that should not be interpreted as a pure
testimony effect without further controls.

Complete local results are under
`results/sycophancy_bias_probe/openai_api/addressee_indexed_endorsement_gpt54nano_20260727/`.
The most useful files are:

- `analysis/report.md`
- `analysis/condition_summary.csv`
- `analysis/comparison_vs_regular.csv`
- `analysis/factorial_contrasts.csv`
- `analysis/question_paired_results.csv`
- `analysis/sycophancy_by_condition.png`
- `audit_summary.json`
