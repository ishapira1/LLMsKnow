# Fixed OpenAI sycophancy development cohort

Use `openai_sycophancy_development_cohort_gpt54nano_v1.jsonl` for future
development experiments with `gpt-5.4-nano-2026-03-17`.

- CommonsenseQA: 1,000 neutral-correct train questions.
- ARC-Challenge: all 959 neutral-correct train questions.
- Selection seed: 5.
- Total paired questions per condition: 1,959.
- Reuse the same question-specific incorrect option in every condition.
- Keep the saved neutral prompt, answer-only instruction, model snapshot, and API
  settings unchanged.
- Change only the experimental condition text and any explicitly manipulated
  system message.
- Treat the cohort as immutable. A changed model, prompt framework, API setting,
  source question, answer options, or incorrect option requires a new version and
  new neutral screening.

The companion JSON specification records the manifest and question-identity
SHA-256 checksums. Run:

```bash
PYTHONPATH=src python3 scripts/freeze_openai_development_cohort.py audit
```

before preparing a new experiment.
