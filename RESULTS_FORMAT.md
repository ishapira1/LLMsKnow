# Results Format And Parsing Guide

This document describes the active artifact contract for:

```bash
python run_sycophancy_bias_probe.py ...
```

It is intentionally narrow: it lists what the current pipeline actually writes, separates core outputs from optional probe artifacts and later derived artifacts, and leaves older legacy files in compatibility mode rather than treating them as part of the active contract.

## Path Rule

Every run lives at:

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
```

- `model_slug`: sanitized model name such as `Qwen_Qwen2_5_7B_Instruct`
- `dataset_dir`: usually `commonsense_qa`, `arc_challenge`, or `all`
- `run_name`: explicit `--run_name`, auto-generated timestamped name, or a `--fresh_run` name

## Core Pipeline Outputs

These are the canonical outputs written by the base pipeline itself.

- `logs/run.log`
  - Human-readable runtime log.
  - Written for every run.
- `logs/sampling_records.jsonl`
  - Canonical raw per-record sampling store.
  - Also used for sampling checkpointing and cache reuse.
- `logs/sampling_manifest.json`
  - Sampling spec, checkpoint state, split stats, and sampling hash.
- `logs/sampling_integrity_summary.json`
  - Post-sampling integrity/compliance summary.
- `logs/warnings.log`
  - Optional warning-only log.
  - Present only if warnings were emitted.
- `logs/warnings_summary.json`
  - Optional structured warning rollup.
  - Present only if warnings were emitted.
- `sampling/sampled_responses.csv`
  - Flat table of sampled rows.
  - Best starting point for pandas analysis.
- `reports/summary.json`
  - Flat summary rows as JSON.
  - This is not the richer nested run summary.
- `reports/summary.csv`
  - CSV version of the flat summary rows.
- `reports/executive_summary.md`
  - Human-readable markdown overview.
- `reports/confusion_matrix_predicted_letter_x_true_letter.csv`
  - Optional multiple-choice confusion matrix export.
  - Written only when that summary is available.
- `probes/probe_scores_by_prompt.csv`
  - Prompt-level probe readout table.
  - Written even for sampling-only runs, though it may be empty or minimal.
- `run_config.json`
  - Resolved run config plus canonical artifact paths and runtime metadata.
- `run_summary.json`
  - Rich nested summary payload.
  - Includes `summary_rows` plus higher-level runtime and reporting sections.
  - `runtime_timing` stores both top-level stage timing and nested probe substage timing.
- `status.json`
  - Lifecycle metadata for the run.
  - The final status write happens last.

## Optional Probe Outputs

These are written only when probe training runs successfully. They are absent for `--sampling_only`.

- `probes/all_probes/manifest.json`
  - Group-level manifest for all trained layer candidates.
- `probes/all_probes/<probe_name>/manifest.json`
  - Per-family manifest for one probe family.
- `probes/all_probes/<probe_name>/layer_XXX/`
  - One directory per trained candidate layer.
- `probes/all_probes/<probe_name>/layer_XXX/model.pkl`
  - Serialized candidate probe model.
- `probes/all_probes/<probe_name>/layer_XXX/metadata.json`
  - Candidate probe metadata.
- `probes/all_probes/<probe_name>/layer_XXX/metrics.json`
  - Candidate probe metrics.
- `probes/all_probes/<probe_name>/layer_XXX/record_membership.jsonl`
  - Candidate probe membership table.
- `probes/chosen_probe/manifest.json`
  - Group-level manifest for the final selected probes.
- `probes/chosen_probe/<probe_name>/`
  - Final selected probe directory for one probe family.
- `probes/chosen_probe/<probe_name>/model.pkl`
  - Final selected probe model.
- `probes/chosen_probe/<probe_name>/metadata.json`
  - Final selected probe metadata.
- `probes/chosen_probe/<probe_name>/metrics.json`
  - Final selected probe metrics.
- `probes/chosen_probe/<probe_name>/record_membership.jsonl`
  - Final selected probe membership table.
- `probes/chosen_probe/<probe_name>/manifest.json`
  - Final selected probe manifest.

## Optional Derived Outputs

These are post-hoc artifacts written by later analysis or backfill scripts. They are supported and organized, but they are not required for a pipeline run to count as complete.

- `analysis/analysis_notebook_status.json`
  - Notebook generation status file.
- `analysis/analysis_<spec>.ipynb`
  - Generated notebook.
- `analysis/plots/*.pdf`
  - Post-hoc plot exports.
- `analysis/tables/*.csv`
  - Post-hoc table exports.
- `analysis/tables/analysis_cell_failures.csv`
  - Notebook cell-level failure log when analysis degrades gracefully.
- `sampling_backfills/<template_type>/...`
  - Derived sampling backfill artifacts.
- `probes/backfills/<probe_name>_all_templates/...`
  - Derived neutral-probe rescoring/backfill artifacts.
- `probes/backfills/<probe_name>_on_<template_type>/...`
  - Derived cross-template probe rescoring/backfill artifacts.

All derived outputs must stay inside the current run directory. The base pipeline does not write them by default.

## Canonical Trees

Sampling-only run:

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
├── logs/
│   ├── run.log
│   ├── sampling_records.jsonl
│   ├── sampling_manifest.json
│   ├── sampling_integrity_summary.json
│   ├── warnings.log                       # optional
│   └── warnings_summary.json              # optional
├── sampling/
│   └── sampled_responses.csv
├── probes/
│   └── probe_scores_by_prompt.csv         # may be empty/minimal
├── reports/
│   ├── summary.json
│   ├── summary.csv
│   ├── executive_summary.md
│   └── confusion_matrix_predicted_letter_x_true_letter.csv   # optional
├── run_config.json
├── run_summary.json
└── status.json
```

Full probe run:

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
├── logs/
├── sampling/
├── probes/
│   ├── probe_scores_by_prompt.csv
│   ├── all_probes/
│   │   ├── manifest.json
│   │   └── <probe_name>/
│   │       ├── manifest.json
│   │       └── layer_XXX/
│   │           ├── model.pkl
│   │           ├── metadata.json
│   │           ├── metrics.json
│   │           └── record_membership.jsonl
│   └── chosen_probe/
│       ├── manifest.json
│       └── <probe_name>/
│           ├── model.pkl
│           ├── metadata.json
│           ├── metrics.json
│           ├── record_membership.jsonl
│           └── manifest.json
├── reports/
├── run_config.json
├── run_summary.json
└── status.json
```

Probe run with post-hoc analysis and backfills:

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
├── logs/
├── sampling/
├── probes/
│   ├── probe_scores_by_prompt.csv
│   ├── all_probes/
│   ├── chosen_probe/
│   └── backfills/                        # optional derived subtree
├── sampling_backfills/                   # optional derived subtree
├── analysis/
│   ├── analysis_notebook_status.json
│   ├── analysis_<spec>.ipynb
│   ├── plots/
│   └── tables/
├── reports/
├── run_config.json
├── run_summary.json
└── status.json
```

## Save Order

The pipeline writes artifacts in this logical order:

1. Create the run directory and configure logging.
2. Write `status.json` as `running`.
3. Persist `logs/sampling_records.jsonl` and `logs/sampling_manifest.json` during sampling.
4. Write `logs/sampling_integrity_summary.json` and optional warning summaries after sampling.
5. Write optional probe directories and manifests if probe training runs.
6. Write final summaries and config:
   - `sampling/sampled_responses.csv`
   - `reports/summary.json`
   - `reports/summary.csv`
   - `run_summary.json`
   - `probes/probe_scores_by_prompt.csv`
   - optional confusion matrix
   - `run_summary.json.runtime_timing` keeps the stage/substage timing hierarchy used during execution
   - `reports/executive_summary.md`
   - `run_config.json`
7. Write final `status.json` as `completed` or `failed` last.

## Parsing Notes

- Start from `logs/sampling_records.jsonl` when you need the canonical raw record stream.
- Start from `sampling/sampled_responses.csv` when you want a flat analysis table.
- Use `reports/summary.csv` or `reports/summary.json` for quick run-level summaries.
- Use `run_summary.json` for the richer nested programmatic payload.
- Use `status.json` plus `logs/sampling_manifest.json` when checking resume state, cache provenance, or completion state.

## Legacy Compatibility

Older runs may still contain or be read from compatibility aliases such as:

- `internal/run_config.json`
- `internal/status.json`
- `internal/logs/run.log`
- `sampled_responses.csv`
- `sampling_manifest.json`

Some older files are still loader-compatible when present, but they are not part of the active write contract and should not be expected in new runs. The current docs intentionally omit those legacy artifact names so they do not look like required modern outputs.
