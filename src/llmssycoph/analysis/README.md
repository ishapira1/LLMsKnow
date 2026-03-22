# Analysis Framework

This package provides a reusable, spec-driven analysis layer for completed runs.

The design goal is to avoid hand-writing notebook logic over and over. Instead:

1. load one run into a canonical `AnalysisContext`
2. build reusable derived dataframes
3. expose small table/plot functions with explicit metadata
4. generate notebooks from a declarative spec

## Output Contract

All generated artifacts must stay inside the current run directory.

- notebook: `run_dir/analysis/analysis_<spec>.ipynb`
- status: `run_dir/analysis/analysis_notebook_status.json`
- plots: `run_dir/analysis/plots/*.pdf`
- tables: `run_dir/analysis/tables/*.csv`
- cell failures: `run_dir/analysis/tables/analysis_cell_failures.csv`

Do not save analysis outputs into source directories or shared global folders.

## Supported Scope

This framework currently supports only multiple-choice runs.

The loader explicitly checks for:

- `task_format == multiple_choice`
- `P(selected)`
- candidate probabilities `P(A)` ... `P(E)`

If a run is not MC-compatible, notebook generation fails safely and writes a failure status file.

## Package Layout

- `core.py`
  - shared context and error classes
- `load.py`
  - run loading, MC validation, canonical output directories
- `dataframes.py`
  - canonical derived dataframes built from saved artifacts
- `functions.py`
  - analysis function registry, metadata, plotting helpers, cell-safe execution wrappers
- `specs.py`
  - notebook section / subsection / cell specs
- `notebook_builder.py`
  - spec-to-notebook generation

## Canonical Dataframes

Add reusable dataframe builders to `dataframes.py` when multiple analyses need the same transformation.

Current examples:

- `build_sampled_responses_df(ctx)`
- `build_neutral_sampled_responses_df(ctx)`
- `build_candidate_probability_long_df(ctx)`
- `build_paired_external_df(ctx)`
- `build_probe_scores_df(ctx)`
- `build_paired_probe_df(ctx)`

Rules:

- builders should return dataframes only
- builders should be deterministic for the same run
- builders should reuse `ctx.cache`
- builders should not write files

## Analysis Functions

Every analysis function should be registered with metadata.

Required metadata:

- `output_kind`: `table` or `plot`
- `description`
- whether probes are required
- whether labels are required
- whether a bias target is required

Function rules:

- accept `ctx` as the first argument
- return exactly one dataframe or one matplotlib figure
- avoid side effects beyond optional artifact saving handled by the framework
- do not assume a notebook environment

## Safe Notebook Execution

Notebook cells should call `safe_display_analysis_operation(...)`, not the raw analysis function.

This wrapper:

- catches per-cell exceptions
- writes a row into `analysis_cell_failures.csv`
- renders a short warning in the notebook
- lets later cells continue running

This is intentional: analysis should degrade gracefully when some artifacts are missing.

## Extending The Notebook System

When adding a new analysis:

1. add or reuse canonical dataframe builders
2. add one analysis function
3. register it with metadata
4. add tests
5. reference it from a notebook spec

If a plot needs special interpretation, put that explanation in the notebook spec as markdown, not in the plotting function itself.

## CLI

Use:

```bash
python scripts/generate_analysis_notebook.py --run_dir <run_dir>
```

This writes a notebook plus a status file, and it should never crash the rest of the run pipeline unless `--raise_on_error` is explicitly requested.
