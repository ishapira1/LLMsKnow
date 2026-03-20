`grading/` owns answer labeling and graded record preparation after generation and before probe fitting.

- `grade.py`: grades one response against the prompt base metadata.
- `records.py`: refreshes cached sample records, computes `T_prompt`, and decides which records are usable for metrics. In strict MC, `T_prompt = P(correct)`.
- `probe_data.py`: prepares neutral and bias-specific train/val/retrain record pools for probe selection.
- `MULTIPLE_CHOICE_DEFINITIONS.md`: source-of-truth terminology for the active `strict_mc` path.

`grading/` does not own question splitting or probe model fitting.
