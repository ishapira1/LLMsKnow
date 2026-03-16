# Hidden Knowledge in Sycophancy: Code Reference

This repository currently centers on one main engineering goal:

Measure whether a model's sycophantic behavior is explained by genuine uncertainty, or by a mismatch between what the model appears to internally support and what it finally says.

This README is the working source of truth for the current code path. It is meant to keep future changes consistent. If the pipeline changes, update this file alongside the code.

## Current Scope

The active workflow in this repo is the sycophancy/hidden-knowledge pipeline driven by:

- `run_sycophancy_bias_probe.py`
- `sycophancy_bias_probe/`
- `script.py` as temporary compatibility glue for older helper imports

Older scripts under `src/` are still in the repo, but they reflect the earlier hallucination pipeline from the original paper codebase. They are not the main reference for the current sycophancy workflow.

## Coding Motivation

From a code and experiment-design perspective, the project is trying to separate two things that are usually mixed together:

- observed behavior: what the model outputs under a prompt
- internal evidence: what the model's hidden states suggest about answer correctness

The pipeline therefore compares:

- a neutral prompt `x`
- a bias-injected prompt `x'`

For both prompts, we sample full raw completions, label their correctness with the current short-answer heuristic, and probe hidden states at the final token of the raw sampled completion. The point is not only to ask "did the model become wrong?", but also "what changed internally when it became wrong?"

In practical terms, the code should let us answer questions like:

- Does bias change output correctness only, or also probe-estimated internal evidence?
- Are some failures low-confidence mistakes?
- Are some failures cases where the model seems internally aligned with the truth but still complies with the biased prompt?

## Main Objects and Notation

The pipeline uses the following conventions throughout the code and outputs:

- `x`: the neutral prompt
- `x'`: the biased prompt
- `y_x`: sampled answer under `x`
- `y_xprime`: sampled answer under `x'`
- `C_x_y`: correctness label for `y_x`
- `C_xprime_yprime`: correctness label for `y_xprime`
- `T_x`: empirical mean correctness over repeated draws from `x`
- `T_xprime`: empirical mean correctness over repeated draws from `x'`
- `probe_x`: probe score for neutral records
- `probe_xprime`: probe score for biased records

For per-sample storage:

- `response_raw`: the full sampled assistant completion
- `response`: the current heuristic short-answer extraction used for correctness labeling

Bias types currently supported by the runner:

- `incorrect_suggestion`
- `doubt_correct`
- `suggest_correct`

## Source-of-Truth Pipeline

The current planned pipeline, and the one the code should continue to preserve unless we intentionally change the design, is:

1. Load the sycophancy dataset from `data/sycophancy-eval/answer.jsonl`.
2. Deduplicate rows and classify each prompt as `neutral` or one of the supported bias types.
3. Build question groups that contain the neutral prompt plus all requested bias variants for the same question.
4. Split by question into train and test sets. We split by question, not by sampled response.
5. For each `(split, question, prompt_type)`, sample `n_draws` model responses.
6. Extract a heuristic short answer from each sampled generation for correctness labeling.
7. Label each sampled answer as correct or incorrect against the gold aliases.
8. Compute `T_prompt`, the empirical mean correctness for each prompt variant over repeated draws.
9. Train a probe on the final-token hidden state of `response_raw` for neutral prompts only, choosing the best layer by validation AUC.
10. Train separate probes for each bias type with the same final-token-of-`response_raw` feature, again selecting the best layer by validation AUC.
11. Score all sampled records with the appropriate trained probe.
12. Pair neutral and biased records into `x / x'` tuples for analysis.
13. Save raw samples, paired tuples, aggregated summaries, probe metadata, and checkpoint metadata.

## Current Implementation Boundaries

Today the main responsibilities are split conceptually like this:

- `run_sycophancy_bias_probe.py`
  Public CLI wrapper that parses args and dispatches into the package pipeline.

- `sycophancy_bias_probe/constants.py`
  Shared experiment constants, bias-template names, and resume-compatibility keys.

- `sycophancy_bias_probe/cli.py`
  CLI argument parsing and lightweight config/environment resolution helpers.

- `sycophancy_bias_probe/dataset.py`
  Prompt classification, deduplication, question grouping, and question-level train/test splitting.

- `sycophancy_bias_probe/runtime.py`
  Run directory management, lock/status handling, and atomic artifact writes.

- `sycophancy_bias_probe/sampling.py`
  Sample-record keys, sampling-spec hashing, cache lookup, resumable sampling, and empirical `T_prompt` computation.

- `sycophancy_bias_probe/outputs.py`
  Construction of `sampled_responses.csv`, `final_tuples.csv`, and `summary_by_question.csv`.

- `sycophancy_bias_probe/probes.py`
  Probe feature extraction across layers from the final token of `response_raw`, layer selection by validation AUC, final probe training, and record scoring.

- `sycophancy_bias_probe/pipeline.py`
  Main orchestration for the end-to-end run: runtime setup, sampling, probing, artifact writing, and status updates.

- `sycophancy_bias_probe/answer_utils.py`
  Gold-answer extraction, answer normalization, correctness checks, and short-answer extraction.

- `sycophancy_bias_probe/model_utils.py`
  Chat formatting, chat encoding, single-draw generation, and batched generation helpers.

- `sycophancy_bias_probe/feature_utils.py`
  Answer/completion-span lookup plus hidden-state and logprob-based feature extraction helpers.

- `script.py`
  Compatibility shim that still owns the legacy dataset I/O helpers and re-exports the moved helper names.

- `jobs/sycophancy_bias_probe/`
  Cluster job launchers for running the pipeline at different scales.

- `notebooks/`
  Downstream analysis over saved run artifacts.

## Artifact Contract

Each completed run writes a directory under `output/sycophancy_bias_probe/<model_slug>/<run_name>/`.

The key files are:

- `sampled_responses.csv`
  One row per sampled response. This is the raw per-draw table and includes both `response_raw` and the heuristic `response`.

- `final_tuples.csv`
  The canonical paired analysis table. Each row joins a neutral sample and a biased sample for the same question, split, bias type, and draw index.

- `summary_by_question.csv`
  Aggregated view over repeated draws for each `(question_id, bias_type, split)`.

- `probe_metadata.json`
  Best selected layer, validation AUC, and saved probe paths for the neutral probe and each bias-specific probe.

- `sampling_records.jsonl`
  Checkpointable raw sampling state used for resuming and cache reuse.

- `sampling_manifest.json`
  Sampling spec hash, completion state, reuse source, and per-split sampling stats.

- `run_config.json`
  The resolved configuration for the run.

- `status.json`
  Run lifecycle state such as `running`, `completed`, or `failed`.

- `probe_models/`
  Serialized sklearn probe models for each layer tested and for the final retrained best layer.

## Important Invariants

These are the most important assumptions to preserve when changing the code:

- Question-level grouping is the unit of comparison. Neutral and biased variants for a question must stay aligned.
- Train/test splitting happens at the question level, not at the sample level.
- `draw_idx` is a bookkeeping key for pairing records. It does not imply shared randomness between `x` and `x'`.
- Probe targets are sampled-answer correctness labels, not the gold answer directly.
- Probe features come from the final token of `response_raw`, not from the shortened `response`.
- The `response` field remains a correctness-labeling convenience only; it is not the probe text.
- `probe_x` is trained on neutral prompt records only.
- `probe_xprime` is trained separately for each bias type.
- `T_x` and `T_xprime` are empirical statistics over repeated samples, not probe outputs.
- Cache reuse is only valid when the sampling specification matches.
- `final_tuples.csv` is the main table downstream analysis should depend on.

## Minimal Run Example

Smoke test:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --smoke_test \
  --run_name smoke_run
```

A larger run follows the same flow, usually by changing:

- `--run_name`
- `--max_questions` or removing `--smoke_test`
- `--n_draws`
- sampling and probe selection settings

## Concrete Refactor Roadmap

The refactor should be done as a sequence of behavior-preserving PRs. The entrypoint stays `run_sycophancy_bias_probe.py` until the last phase. Every PR must update this README if any module boundary, artifact contract, or invariant changes.

Current status:

- PR 1 is implemented: constants and CLI/config helpers now live under `sycophancy_bias_probe/`.
- PR 2 is implemented: dataset normalization, prompt classification, grouping, and question-level splitting now live under `sycophancy_bias_probe/dataset.py`.
- PR 3 is implemented: run directories, resume checks, locks, status updates, and atomic writes now live under `sycophancy_bias_probe/runtime.py`.
- PR 4 is implemented: sample-record keys, sampling-spec hashing, cache reuse, resumable sampling, and empirical `T_prompt` now live under `sycophancy_bias_probe/sampling.py`.
- PR 5 is implemented: tuple rows, sample tables, and question-level summary tables now live under `sycophancy_bias_probe/outputs.py`.
- PR 6 is implemented: probe feature extraction, layer search, probe training, and scoring now live under `sycophancy_bias_probe/probes.py`.
- PR 7 is implemented: shared answer, generation, and feature helpers now live under `sycophancy_bias_probe/answer_utils.py`, `sycophancy_bias_probe/model_utils.py`, and `sycophancy_bias_probe/feature_utils.py`, while `script.py` remains as compatibility glue for older imports and legacy I/O helpers.
- PR 8 is implemented: the runner orchestration now lives in `sycophancy_bias_probe/pipeline.py`, and `run_sycophancy_bias_probe.py` is a thin public wrapper.
- The remaining phase below is optional cleanup work.

The proposed internal package layout is:

- `sycophancy_bias_probe/__init__.py`
- `sycophancy_bias_probe/constants.py`
- `sycophancy_bias_probe/cli.py`
- `sycophancy_bias_probe/dataset.py`
- `sycophancy_bias_probe/runtime.py`
- `sycophancy_bias_probe/sampling.py`
- `sycophancy_bias_probe/probes.py`
- `sycophancy_bias_probe/outputs.py`
- `sycophancy_bias_probe/answer_utils.py`
- `sycophancy_bias_probe/model_utils.py`
- `sycophancy_bias_probe/feature_utils.py`
- `sycophancy_bias_probe/pipeline.py`
- `sycophancy_bias_probe/pipeline.py`

Recommended test/support files to add during the refactor:

- `tests/test_artifact_contract.py`
- `tests/test_dataset_contract.py`
- `tests/test_runtime_contract.py`
- `tests/test_sampling_contract.py`
- `tests/test_probe_contract.py`
- `tests/test_output_contract.py`
- `tests/test_script_compatibility.py`
- `tests/test_pipeline_contract.py`
- `tests/fixtures/sycophancy_rows_sample.jsonl`
- `tests/compare_run_artifacts.py`

### PR 1: Extract constants and CLI/config helpers

New files:

- `sycophancy_bias_probe/constants.py`
- `sycophancy_bias_probe/cli.py`

Move from `run_sycophancy_bias_probe.py`:

- `NEUTRAL_TEMPLATE`
- `BIAS_TEMPLATE_TO_TYPE`
- `ALL_BIAS_TYPES`
- `RESUME_COMPAT_KEYS`
- `parse_args`
- `resolve_bias_types`
- `resolve_device`
- `resolve_hf_cache_dir`
- `load_env_file`

Keep unchanged in the runner:

- all runtime behavior
- all output paths
- the `main()` control flow

README update required:

- add the new package layout once these files exist
- keep the pipeline and artifact sections unchanged

Acceptance checks:

- `python run_sycophancy_bias_probe.py --help`
- `python -c "from sycophancy_bias_probe.cli import parse_args"`
- runner still produces the same `run_config.json` keys for the same CLI flags

### PR 2: Extract dataset normalization and split logic

New files:

- `sycophancy_bias_probe/dataset.py`
- `tests/test_dataset_contract.py`
- `tests/fixtures/sycophancy_rows_sample.jsonl`

Move from `run_sycophancy_bias_probe.py`:

- `_as_prompt_text`
- `_question_key`
- `_template_type`
- `deduplicate_rows`
- `build_question_groups`
- `split_groups`

README update required:

- document `dataset.py` under implementation boundaries
- keep the question-level split invariant explicit

Acceptance checks:

- `pytest tests/test_dataset_contract.py -q`
- for the same sampled fixture, the old and new code paths produce the same prompt types, question groups, and train/test question IDs
- the runner still logs the same `raw_rows`, `dedup_rows`, and `valid_groups` counts on the smoke configuration

### PR 3: Extract runtime, locking, and atomic write utilities

New files:

- `sycophancy_bias_probe/runtime.py`
- `tests/test_runtime_contract.py`

Move from `run_sycophancy_bias_probe.py`:

- `_utc_now_iso`
- `model_slug`
- `_build_default_run_name`
- `make_run_dir`
- `assert_resume_compatible`
- `run_lock_path`
- `_is_pid_alive`
- `acquire_run_lock`
- `release_run_lock`
- `write_json_atomic`
- `write_jsonl_atomic`
- `write_csv_atomic`
- `write_pickle_atomic`
- `write_run_status`

README update required:

- list `runtime.py` in the implementation boundaries section
- keep the run directory layout unchanged in the artifact contract

Acceptance checks:

- `pytest tests/test_runtime_contract.py -q`
- creating a run dir with the same `--run_name` still enforces resume compatibility
- `status.json`, `.run.lock`, `sampling_manifest.json`, and final output files keep the same names and locations

### PR 4: Extract sampling keys, manifests, cache reuse, and empirical T

New files:

- `sycophancy_bias_probe/sampling.py`
- `tests/test_sampling_contract.py`

Move from `run_sycophancy_bias_probe.py`:

- `sample_record_key_values`
- `sample_record_key`
- `sort_sample_records`
- `enumerate_expected_sample_keys`
- `normalize_sample_records`
- `build_sampling_spec`
- `sampling_spec_hash`
- `load_sampling_cache_candidate`
- `sample_records_for_groups`
- `add_empirical_t`

README update required:

- keep `sampling_records.jsonl` and `sampling_manifest.json` documented exactly
- keep the cache reuse rule explicit: reuse is valid only when the sampling spec matches

Acceptance checks:

- `pytest tests/test_sampling_contract.py -q`
- for a fixed config and fixed question IDs, `sampling_hash` is unchanged
- on a cached smoke run, `expected_records`, `n_records`, `is_complete`, `train_stats`, and `test_stats` remain unchanged
- `T_x` and `T_xprime` values in downstream outputs remain unchanged for the same cached records

### PR 5: Extract output table construction

New files:

- `sycophancy_bias_probe/outputs.py`
- `tests/test_output_contract.py`

Move from `run_sycophancy_bias_probe.py`:

- `build_tuple_rows`
- `to_samples_df`

README update required:

- keep `final_tuples.csv` marked as the main downstream contract
- keep the definitions of `sampled_responses.csv`, `final_tuples.csv`, and `summary_by_question.csv` aligned with the code

Acceptance checks:

- `pytest tests/test_output_contract.py -q`
- on the smoke run, the columns in `sampled_responses.csv`, `final_tuples.csv`, and `summary_by_question.csv` are unchanged
- row counts and grouping keys are unchanged

### PR 6: Extract probe selection, training, and scoring

New files:

- `sycophancy_bias_probe/probes.py`
- `tests/test_probe_contract.py`

Move from `run_sycophancy_bias_probe.py`:

- `_find_sublist`
- `get_hidden_feature_all_layers_for_answer`
- `maybe_subsample`
- `select_best_layer_by_auc`
- `train_probe_for_layer`
- `score_records_with_probe`

README update required:

- keep the distinction between `probe_x` and `probe_xprime`
- keep the invariant that the neutral probe is trained on neutral records only, and bias probes are trained separately per bias type

Acceptance checks:

- `pytest tests/test_probe_contract.py -q`
- on a cached smoke run, `probe_metadata.json` keeps the same keys and selected layers
- probe score columns remain numerically equal or within a documented tolerance
- saved model filenames under `probe_models/` remain unchanged

### PR 7: Split `script.py` into stable shared helper modules

New files:

- `sycophancy_bias_probe/answer_utils.py`
- `sycophancy_bias_probe/model_utils.py`
- `sycophancy_bias_probe/feature_utils.py`

Move from `script.py`:

- to `answer_utils.py`
  - `extract_gold_answers_from_base`
  - `normalize_answer`
  - `is_correct_short_answer`
  - `extract_short_answer_from_generation`

- to `model_utils.py`
  - `to_hf_chat`
  - `encode_chat`
  - `generate_one`
  - `_clear_device_cache`
  - `_should_fallback_to_sequential`
  - `generate_many`

- to `feature_utils.py`
  - `_find_sublist`
  - `get_hidden_feature_for_answer`
  - `score_logprob_answer`
  - `score_p_true`

Keep temporarily in `script.py` as a compatibility shim:

- `ensure_sycophancy_eval_cached`
- `read_jsonl`
- import/re-export wrappers for moved helpers until all callers are updated

README update required:

- replace references to `script.py` as the main helper file with the new helper modules
- keep a short note that `script.py` is temporary compatibility glue until removed

Acceptance checks:

- `python -c "from script import generate_many, get_hidden_feature_for_answer"`
- `python -c "from sycophancy_bias_probe.model_utils import generate_many"`
- the runner and existing notebooks still import successfully after the split

### PR 8: Introduce a dedicated pipeline module and slim the runner

New files:

- `sycophancy_bias_probe/pipeline.py`

Move from `run_sycophancy_bias_probe.py`:

- the body of `main()` into a callable such as `run_pipeline(args)`
- the nested `persist_sampling_state` helper

Keep in `run_sycophancy_bias_probe.py`:

- CLI entrypoint only
- a thin `main()` wrapper that parses args and calls `run_pipeline(args)`

README update required:

- update the implementation boundaries section so `pipeline.py` is the orchestrator and `run_sycophancy_bias_probe.py` is the public wrapper

Acceptance checks:

- `python run_sycophancy_bias_probe.py --help`
- a cached smoke run completes successfully through the public entrypoint
- artifact names, schemas, and metadata keys remain unchanged

### PR 9: Optional cleanup once downstream callers are migrated

Possible files to remove or simplify:

- `script.py`, if no remaining callers depend on it

Possible cleanup actions:

- remove temporary compatibility re-exports
- move any remaining legacy-only helpers out of the current path
- tighten imports so new code only uses the `sycophancy_bias_probe/` package

README update required:

- remove the compatibility note only after the shim is actually gone

Acceptance checks:

- all repo-local imports use the new module layout
- runner, jobs, and notebooks still load with the updated imports

## Refactor-Wide Test Policy

Every refactor PR should include three layers of checking:

1. Module-level tests for the code moved in that PR.
2. A lightweight contract check over artifact schemas and metadata keys.
3. A smoke run comparison against a known baseline run.

The baseline comparison should focus on:

- output file presence
- CSV column names
- JSON metadata keys
- question IDs and split membership
- selected probe layers
- manifest structure

It is acceptable for timestamps, lock metadata, and run directory names to differ.

## Refactor-Wide README Policy

For every PR in the roadmap:

- update this README in the same PR
- keep the artifact contract current
- keep the invariants current
- add or remove modules from the implementation boundaries section as soon as the code changes

If a code change is not reflected here, the refactor is incomplete.
