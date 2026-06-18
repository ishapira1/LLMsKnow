# Results Format And Parsing Guide

This document describes the active **results layout v2** for:

```bash
python run_sycophancy_bias_probe.py ...
```

The v2 contract is designed for two use cases at once:

- a **nested tree** that is easy to browse manually
- a **query layer** that is easy to parse programmatically without globbing

## Path Rule

Every run lives at:

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
```

- `model_slug`: sanitized model name such as `Qwen_Qwen2_5B_Instruct`
- `dataset_dir`: usually `commonsense_qa`, `arc_challenge`, or `all`
- `run_name`: explicit `--run_name`, auto-generated timestamped name, or a `--fresh_run` name

## Canonical Tree

```text
results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/
├── meta/
│   ├── run_manifest.json
│   ├── run_config.json
│   ├── run_summary.json
│   └── status.json
├── runtime/
│   └── logs/
│       ├── run.log
│       ├── warnings.log                       # optional
│       └── warnings_summary.json              # optional
├── sampling/
│   ├── raw/
│   │   ├── sampling_records.jsonl
│   │   ├── sampling_manifest.json
│   │   └── sampling_integrity_summary.json
│   └── flat/
│       └── sampled_responses.csv
├── evaluation/
│   └── run/
│       ├── summary.json
│       ├── summary.csv
│       ├── executive_summary.md
│       └── confusion_matrix_predicted_letter_x_true_letter.csv   # optional
├── probes/
│   ├── candidates/
│   │   ├── manifest.json
│   │   └── families/
│   │       └── <probe_name>/
│   │           ├── manifest.json
│   │           └── layers/
│   │               └── layer_XXX/
│   │                   ├── model.pkl
│   │                   ├── metadata.json
│   │                   ├── metrics.json
│   │                   └── record_membership.jsonl
│   ├── chosen/
│   │   ├── manifest.json
│   │   └── families/
│   │       └── <probe_name>/
│   │           ├── model.pkl
│   │           ├── metadata.json
│   │           ├── metrics.json
│   │           ├── record_membership.jsonl
│   │           ├── manifest.json
│   │           └── evaluation/
│   │               ├── cross_family/
│   │               │   ├── manifest.json
│   │               │   ├── all_metrics.json
│   │               │   ├── all_metrics.csv
│   │               │   └── targets/
│   │               │       └── <target_template_type>/
│   │               │           ├── metrics.json
│   │               │           └── metrics.csv
│   │               └── movement/
│   │                   ├── manifest.json
│   │                   ├── coverage.json
│   │                   ├── all_items.jsonl
│   │                   ├── all_summary.json
│   │                   ├── all_summary.csv
│   │                   └── targets/
│   │                       ├── prompt_family/
│   │                       │   └── <target_template_type>/
│   │                       │       ├── items.jsonl
│   │                       │       ├── summary.json
│   │                       │       └── summary.csv
│   │                       └── paraphrase/
│   │                           └── same_family/
│   │                               ├── items.jsonl
│   │                               ├── summary.json
│   │                               └── summary.csv
│   └── backfills/                                  # optional derived subtree
├── query/
│   ├── artifact_catalog.jsonl
│   ├── probe_scores_by_prompt.csv
│   ├── chosen_probe_registry.csv
│   ├── chosen_probe_metrics.csv
│   ├── chosen_probe_cross_family_metrics.csv
│   ├── chosen_probe_movement_summary.csv
│   ├── chosen_probe_movement_items.jsonl
│   └── paraphrase_coverage.csv
├── sampling_backfills/                             # optional derived subtree
└── analysis/                                       # optional derived subtree
    ├── analysis_notebook_status.json
    ├── analysis_<spec>.ipynb
    ├── plots/
    └── tables/
```

## Source Of Truth

There are two canonical entrypoints:

1. `meta/run_manifest.json`
   - the root navigation manifest
   - use this when you need to discover where artifacts live

2. `query/`
   - the direct-query layer
   - use this when you want metrics or movement summaries without traversing probe-local files

The intended rule is:

- for **run-wide questions**, start in `query/`
- for **probe-specific deep dives**, start in `probes/chosen/families/<probe_name>/`
- for **artifact discovery**, start in `meta/run_manifest.json`

## Core Pipeline Outputs

These are written by the base pipeline itself.

- `meta/run_manifest.json`
  - root navigation manifest for the run
- `meta/run_config.json`
  - resolved run config plus artifact paths and runtime metadata
- `meta/run_summary.json`
  - rich nested summary payload
  - includes `runtime_timing` for stage and substage timing
- `meta/status.json`
  - lifecycle metadata for the run
- `runtime/logs/run.log`
  - human-readable runtime log
- `sampling/raw/sampling_records.jsonl`
  - canonical raw per-record sampling store
- `sampling/raw/sampling_manifest.json`
  - sampling spec, checkpoint state, split stats, and sampling hash
- `sampling/raw/sampling_integrity_summary.json`
  - post-sampling integrity/compliance summary
- `sampling/flat/sampled_responses.csv`
  - flat sampled-response table
- `evaluation/run/summary.json`
  - flat summary rows as JSON
- `evaluation/run/summary.csv`
  - CSV version of the flat summary rows
- `evaluation/run/executive_summary.md`
  - human-readable markdown overview
- `query/probe_scores_by_prompt.csv`
  - prompt-level probe readout table

## Optional Probe Outputs

These are written only when probe training runs successfully. They are absent for `--sampling_only`.

- `probes/candidates/`
  - all trained layer candidates
- `probes/chosen/`
  - final chosen probe artifacts
- `probes/chosen/families/<probe_name>/evaluation/cross_family/`
  - cross-family evaluation files for one chosen probe
- `probes/chosen/families/<probe_name>/evaluation/movement/`
  - activation-movement evaluation files for one chosen probe

The movement subtree includes both:

- full item-level rows in `all_items.jsonl`
- grouped summaries in `all_summary.json` and `all_summary.csv`

It also breaks movement out by target:

- `targets/prompt_family/<target_template_type>/`
- `targets/paraphrase/same_family/`

## Query Tables

The `query/` directory contains stable denormalized tables for direct questions.

- `artifact_catalog.jsonl`
  - generic artifact inventory with paths and schema versions
- `chosen_probe_registry.csv`
  - one row per chosen probe family
- `chosen_probe_metrics.csv`
  - one row per chosen probe family with own-family metrics
- `chosen_probe_cross_family_metrics.csv`
  - one row per chosen probe family × target family
- `chosen_probe_movement_summary.csv`
  - one row per chosen probe family × target change
- `chosen_probe_movement_items.jsonl`
  - one row per movement comparison
- `paraphrase_coverage.csv`
  - one row per chosen probe family with paraphrase-coverage counts

These query files are the preferred way to answer questions such as:

- “What is the chosen probe AUC for `probe_bias_incorrect_suggestion_strong`?”
- “What is the average movement geometry for `probe_no_bias` when the target family is `strong`?”
- “How many paraphrase comparisons were skipped because of invalid paraphrases?”

## Optional Derived Outputs

These are post-hoc artifacts written by later analysis or backfill scripts. They are supported and organized, but they are not required for a pipeline run to count as complete.

- `analysis/analysis_notebook_status.json`
- `analysis/analysis_<spec>.ipynb`
- `analysis/plots/*.pdf`
- `analysis/tables/*.csv`
- `sampling_backfills/<template_type>/...`
- `probes/backfills/<probe_name>_all_templates/...`
- `probes/backfills/<probe_name>_on_<template_type>/...`

## Save Order

The pipeline writes artifacts in this logical order:

1. Create the run directory and configure logging.
2. Write `meta/status.json` as `running`.
3. Persist sampling artifacts under `sampling/raw/` during sampling.
4. Write `sampling/raw/sampling_integrity_summary.json` and optional warning summaries after sampling.
5. Write optional probe directories and manifests if probe training runs.
6. Write final flat outputs:
   - `sampling/flat/sampled_responses.csv`
   - `evaluation/run/summary.json`
   - `evaluation/run/summary.csv`
   - `meta/run_summary.json`
   - `query/probe_scores_by_prompt.csv`
   - optional confusion matrix
   - `evaluation/run/executive_summary.md`
   - `meta/run_config.json`
7. Write query tables under `query/`.
8. Write `meta/run_manifest.json`.
9. Write final `meta/status.json` as `completed` or `failed` last.

## Parsing Notes

- Start from `meta/run_manifest.json` when you need artifact discovery.
- Start from `query/` when you want direct metric lookup.
- Start from `sampling/raw/sampling_records.jsonl` when you need the canonical raw record stream.
- Start from `sampling/flat/sampled_responses.csv` when you want the flat model-output table.
- Use probe-local `evaluation/` subtrees when you want one chosen probe in detail.

## Legacy Compatibility

Older runs may still contain legacy layouts such as:

- root-level `run_config.json`, `run_summary.json`, `status.json`
- `logs/`
- `reports/`
- `sampling/sampled_responses.csv`
- `probes/all_probes/`
- `probes/chosen_probe/`

Loaders may still resolve some of these for backward compatibility, but they are **not** part of the active v2 write contract for new runs.
