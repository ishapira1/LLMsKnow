# Results Format And Parsing Guide

This document describes the artifacts written by the active sycophancy pipeline and how to parse them for downstream analysis.

The public entrypoint is:

```bash
python run_sycophancy_bias_probe.py ...
```

## Run directory layout

Each run writes to:

```text
output/sycophancy_bias_probe/<model_slug>/<run_name>/
```

Typical contents:

```text
run.log
sampling_records.jsonl
sampling_manifest.json
sampled_responses.csv
final_tuples.csv
summary_by_question.csv
probe_metadata.json
probe_models/
run_config.json
status.json
```

## Which file is canonical for what

- `sampling_records.jsonl`
  - Canonical raw sampled-response store.
  - One JSON object per sampled completion.
  - This is the file used for sampling checkpointing and cache reuse.
- `sampled_responses.csv`
  - Flat table version of the sampled records.
  - Best starting point for pandas-based analysis.
- `final_tuples.csv`
  - Pair table that matches neutral and biased prompts for the same `(split, question_id, draw_idx)`.
  - Only includes rows where both sides are usable for metrics.
- `summary_by_question.csv`
  - Question-level aggregation over repeated draws.
- `probe_metadata.json`
  - Probe-selection and saved-model metadata.
- `run_config.json`
  - Full resolved CLI config for the run.
- `sampling_manifest.json`
  - The exact sampling/cache spec plus checkpoint status.
- `run.log`
  - Human-readable runtime log for the run.
  - Mirrors the stage and progress messages printed during execution.

If you want the complete record of model generations, start from `sampling_records.jsonl`.

## Cache reuse rules

The pipeline reuses previously generated model responses only when the `sampling_hash` matches exactly.

That hash is built from the sampling spec recorded in `sampling_manifest.json`, including:

- `sampling_spec_version`
- `model`
- `benchmark_source`
- `input_jsonl`
- `dataset_name`
- `ays_mc_datasets`
- `sycophancy_repo`
- `bias_types`
- `seed`
- `n_draws`
- `sample_batch_size`
- `temperature`
- `top_p`
- `max_new_tokens`
- `test_frac`
- `probe_val_frac`
- `split_seed`
- `max_questions`
- `smoke_test`
- `smoke_questions`
- `train_question_ids`
- `val_question_ids`
- `test_question_ids`
- `expected_train_records`
- `expected_val_records`
- `expected_test_records`

Implications:

- If these match, the pipeline will reuse sampled responses instead of generating them again.
- If any of these differ, the sampling hash changes and the old responses are not reused.
- `--no_reuse_sampling_cache` disables reuse even when the hash matches.

Probe hyperparameters do not affect the sampling hash, because they do not change the base-model responses.

## Per-sample schema

`sampling_records.jsonl` and `sampled_responses.csv` contain one record per sampled completion.

Important fields:

- `record_id`: stable row id within the run
- `split`: `train`, `val`, or `test`
- `question_id`: question-group id such as `q_17`
- `dataset`: source dataset from `base.dataset`, for example `trivia_qa` or `truthful_qa`
- `template_type`: `neutral` or a bias type like `incorrect_suggestion`
- `draw_idx`: repeated-sampling index for the same prompt
- `task_format`: empty for the original freeform benchmark, or `multiple_choice` for AYS-derived MC rows
- `correct_letter`, `letters`, `answer_options`, `answers_list`: preserved MC metadata when the source row came from the AYS-derived benchmark
- `prompt_messages`: original chat-message structure
- `prompt_text`: flattened prompt text
- `prompt_template`: template string used to build the prompt
- `question`, `correct_answer`, `incorrect_answer`, `gold_answers`
  - `question` is the original `base.question`
- `response_raw`: full raw model completion
- `response`: parsed short answer used for grading
- `correctness`: `1`, `0`, or null for ambiguous/unusable rows
- `grading_status`: grading label
- `grading_reason`: why the row was graded or marked ambiguous
- `usable_for_metrics`: whether the row is eligible for probe training and accuracy metrics
- `T_prompt`: empirical prompt accuracy for this `(split, question_id, template_type)`
- `probe_x`, `probe_xprime`: probe scores after probe training/scoring finishes

## How response processing works

The grading path is:

1. Generate `response_raw`
2. Extract a short answer into `response`
3. Compare `response` against `gold_answers`
4. Store:
   - `correctness = 1` for correct
   - `correctness = 0` for incorrect
   - `correctness = null` for ambiguous/unusable
5. Mark `usable_for_metrics`

Ambiguous rows are preserved in the raw outputs but excluded from:

- prompt-accuracy estimation
- tuple construction
- probe training

When `benchmark_source=ays_mc_single_turn`, the pipeline first materializes `are_you_sure.jsonl` multiple-choice rows into synthetic `answer.jsonl`-style prompts using the existing prompt templates. Grading then accepts either the correct option letter or the correct answer text.

## Pair-level schema

`final_tuples.csv` matches a neutral prompt and a biased prompt for the same:

- `split`
- `question_id`
- `draw_idx`

Key columns:

- `dataset`
- `prompt_template_x`, `prompt_template_xprime`
- `bias_type`
- `prompt_x`, `prompt_with_bias`
- `y_x`, `y_xprime`
- `C_x_y`, `C_xprime_yprime`
- `T_x`, `T_xprime`
- `probe_x`, `probe_xprime`

This is the main analysis table for comparing neutral and biased behavior.

## Question-level summary schema

`summary_by_question.csv` aggregates `final_tuples.csv` by:

- `model_name`
- `split`
- `question_id`
- `bias_type`

Key columns:

- `dataset`
- `prompt_template_x`, `prompt_template_xprime`
- `mean_C_x`
- `mean_C_xprime`
- `mean_probe_x`
- `mean_probe_xprime`
- `n_draws`

Use this table when you want one row per question and bias type rather than one row per sampled pair.

## Probe metadata

`probe_metadata.json` stores, for the neutral probe and each bias-specific probe:

- selected best layer
- dev AUC for the selected layer
- AUC per scanned layer
- feature-source description
- saved layer-scan models
- saved final retrained model

The serialized sklearn models live under `probe_models/`.

## Recommended parsing order

Use one of these flows:

1. Full-fidelity analysis
   - Read `sampling_records.jsonl`
   - Filter by `usable_for_metrics` as needed
   - Build any custom aggregates yourself

2. Standard prompt-vs-bias analysis
   - Read `final_tuples.csv`
   - Compute transitions, deltas, and sycophancy-style metrics there

3. Question-level summaries
   - Read `summary_by_question.csv`
   - Group by `split` and `bias_type`

## Minimal parsing example

```python
import json
from pathlib import Path

import pandas as pd

run_dir = Path("output/sycophancy_bias_probe/<model_slug>/<run_name>")

sampled = pd.read_csv(run_dir / "sampled_responses.csv")
tuples = pd.read_csv(run_dir / "final_tuples.csv")
summary = pd.read_csv(run_dir / "summary_by_question.csv")
probe_meta = json.loads((run_dir / "probe_metadata.json").read_text())
manifest = json.loads((run_dir / "sampling_manifest.json").read_text())

# Example: per-bias accuracy on paired rows
paired_accuracy = (
    tuples.groupby(["split", "bias_type"], as_index=False)
    .agg(
        neutral_accuracy=("C_x_y", "mean"),
        biased_accuracy=("C_xprime_yprime", "mean"),
        mean_probe_x=("probe_x", "mean"),
        mean_probe_xprime=("probe_xprime", "mean"),
        n_pairs=("draw_idx", "size"),
    )
)

# Example: "harmful flip" rate on paired rows
tuples["harmful_flip"] = ((tuples["C_x_y"] == 1) & (tuples["C_xprime_yprime"] == 0)).astype(int)
harmful_flip_rate = (
    tuples.groupby(["split", "bias_type"], as_index=False)
    .agg(harmful_flip_rate=("harmful_flip", "mean"))
)

print(paired_accuracy)
print(harmful_flip_rate)
print(manifest["sampling_spec"])
print(probe_meta.keys())
```

## Practical notes

- For exact raw generations, prefer `sampling_records.jsonl` over the CSVs.
- For custom transitions and sycophancy metrics, prefer `final_tuples.csv`.
- For fast overviews, prefer `summary_by_question.csv`.
- If you are comparing runs, always inspect both `run_config.json` and `sampling_manifest.json`.
