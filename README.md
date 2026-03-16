# Hidden Knowledge in Sycophancy

This repository contains the code for measuring whether a model's sycophantic behavior reflects genuine uncertainty or a mismatch between what its hidden states support and what it ultimately says.

The active code path samples responses to neutral and bias-injected prompts, labels answer correctness, trains linear probes on hidden states, and compares internal evidence against observed behavior.

## What the pipeline does

For each question, the pipeline builds:

- a neutral prompt `x`
- one or more biased prompt variants `x'`

It then:

1. Splits question groups into question-level `train`, `val`, and `test` sets.
2. Samples multiple responses from each prompt in each split.
3. Extracts a short answer for evaluation.
4. Labels each sampled answer as correct, incorrect, or ambiguous.
5. Estimates empirical correctness for each prompt variant over repeated draws.
6. Trains candidate probes on `train`, choosing the best layer by AUC on `val`.
7. Retrains the selected layer on `train + val`.
8. Scores sampled records with the retrained probes.
9. Produces paired neutral vs. biased records for downstream analysis.

The main question is whether bias changes only the model's output, or also changes the internal evidence available in its representations.

## Repository layout

- `run_sycophancy_bias_probe.py`: public entrypoint for the current pipeline
- `sycophancy_bias_probe/`: main package for dataset prep, sampling, probes, outputs, and runtime helpers
- `sycophancy_bias_probe/correctness.py`: source-of-truth answer parsing and correctness grading logic
- `RESULTS_FORMAT.md`: artifact layout, cache rules, and parsing guide for run outputs
- `jobs/sycophancy_bias_probe/`: SLURM job scripts for cluster runs
- `notebooks/`: downstream analysis notebooks
- `data/`: local datasets used by the experiments
- `src/`: older scripts from the earlier codebase; retained for reference, not the main workflow
- `script.py`: legacy compatibility and shared helper functions still used by the current runner

## Setup

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for non-trivial runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline loads Hugging Face models and uses the sycophancy evaluation data in `data/sycophancy-eval/`. If those files are missing, the runner can fetch them from `meg-tong/sycophancy-eval`.

## Quick start

Run a smoke test:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --smoke_test \
  --run_name smoke_run
```

Run a larger experiment:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --run_name main_run \
  --n_draws 8 \
  --max_questions 200 \
  --sample_batch_size 4 \
  --probe_layer_min 1 \
  --probe_layer_max 32
```

Run only one source dataset such as `truthful_qa`:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --run_name truthful_only_run \
  --dataset_name truthful_qa
```

Run the AYS-derived single-turn MC benchmark on the recommended starting slices:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --benchmark_source ays_mc_single_turn \
  --input_jsonl are_you_sure.jsonl \
  --ays_mc_datasets truthful_qa_mc,aqua_mc \
  --run_name ays_mc_truthful_aqua_run
```

Show all options:

```bash
python run_sycophancy_bias_probe.py --help
```

## Key configuration

Important flags:

- `--model`: Hugging Face model name
- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--benchmark_source`: `answer_json` for the existing `answer.jsonl` benchmark, or `ays_mc_single_turn` to derive a new single-turn benchmark from AYS multiple-choice source rows
- `--input_jsonl`: `answer.jsonl` for the original pipeline, or `are_you_sure.jsonl` when using `--benchmark_source ays_mc_single_turn`
- `--bias_types`: comma-separated subset of `incorrect_suggestion`, `doubt_correct`, `suggest_correct`
- `--dataset_name` / `--dataset_type`: source dataset from `base.dataset` to keep, or `all` to use every dataset
- `--ays_mc_datasets`: comma-separated AYS source datasets to derive in `ays_mc_single_turn` mode; default is `truthful_qa_mc,aqua_mc`
- `--n_draws`: number of sampled completions per prompt
- `--max_questions`: limit the number of question groups
- `--test_frac`: fraction of questions reserved for the held-out test split
- `--val_frac` / `--probe_val_frac`: fraction of the non-test questions reserved for validation during probe layer selection
- `--sample_batch_size`: generation batch size
- `--hf_cache_dir`: cache directory for model and tokenizer files
- `--out_dir`: root output directory
- `--run_name`: explicit name for the run directory

## Outputs

Each run writes to:

`output/sycophancy_bias_probe/<model_slug>/<run_name>/`

Main artifacts:

- `sampled_responses.csv`: one row per sampled completion, including the source `dataset`, original `question`, `prompt_template`, split membership, grading result, and for MC-derived runs the preserved choice metadata (`correct_letter`, `letters`, `answer_options`, `answers_list`)
- `final_tuples.csv`: paired neutral and biased records for the same question and draw index, including `dataset` plus both prompt templates, after dropping ambiguous samples
- `summary_by_question.csv`: question-level aggregates across repeated draws, grouped by split, with `dataset` and prompt-template provenance retained
- `probe_metadata.json`: selected layers, validation metrics, and saved probe paths
- `sampling_records.jsonl`: resumable per-sample checkpoint state
- `sampling_manifest.json`: sampling spec and checkpoint metadata
- `run_config.json`: resolved run configuration
- `status.json`: run lifecycle state
- `probe_models/`: serialized sklearn probe models

`final_tuples.csv` is the main table intended for downstream analysis.

For artifact schemas and parsing guidance, see `RESULTS_FORMAT.md`.

## Current implementation notes

- Train/validation/test splitting is done at the question level, not the sample level.
- Sampled answers are parsed into short answers and labeled as `correct`, `incorrect`, or `ambiguous`.
- Ambiguous or unparseable samples are preserved in raw outputs but excluded from paired correctness metrics and probe training.
- Probe targets are sampled-answer correctness labels for graded samples only.
- Probe features come from the final token of `response_raw`.
- Neutral and bias-specific probes are trained separately.
- Probe layer selection is done by validation AUC on the held-out `val` split.
- After selecting the best layer, the final probe is retrained on `train + val` before scoring records.
- The `test` split stays untouched during layer selection and is the clean held-out evaluation split.
- Sampling checkpoints can be reused when the sampling specification matches.

## Cluster runs

Cluster launch scripts live in `jobs/sycophancy_bias_probe/`.

Examples:

```bash
sbatch jobs/sycophancy_bias_probe/fast_dirty.sbatch
sbatch jobs/sycophancy_bias_probe/medium.sbatch
sbatch jobs/sycophancy_bias_probe/full.sbatch
```

Those scripts assume a specific lab environment and cache setup, so they should be treated as templates unless you are running in the same environment.

## Legacy code

The repository still contains older scripts under `src/` from the earlier hallucination-focused codebase. They are not the source of truth for the current sycophancy pipeline.
