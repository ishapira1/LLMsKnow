# Hidden Knowledge in Sycophancy

This repository contains the code for measuring whether a model's sycophantic behavior reflects genuine uncertainty or a mismatch between what its hidden states support and what it ultimately says.

The active code path samples or scores responses to neutral and bias-injected prompts, labels answer correctness, trains linear probes on hidden states, and compares internal evidence against observed behavior.

## What the pipeline does

For each question, the pipeline builds:

- a neutral prompt `x`
- one or more biased prompt variants `x'`

It then:

1. Splits question groups into question-level `train`, `val`, and `test` sets.
2. Produces one or more response records for each prompt in each split.
3. Extracts a short answer for evaluation.
4. Labels each sampled answer as correct, incorrect, or ambiguous.
5. Computes prompt-level `T_prompt`: empirical mean correctness over repeated draws in generation-based paths, and `P(correct)` in strict MC.
6. Trains candidate probes on `train`, choosing the best layer by AUC on `val`.
   For strict MC, probe examples are explicit teacher-forced `(prompt, answer_choice)` rows and are probability-weighted by default.
7. Retrains the selected layer on `train + val`.
8. Scores sampled records with the retrained probes and, for strict MC, also scores every `(prompt, answer_choice)` pair.
9. Produces paired neutral vs. biased records for downstream analysis.

The main question is whether bias changes only the model's output, or also changes the internal evidence available in its representations.

## Repository layout

- `run_sycophancy_bias_probe.py`: thin public wrapper for the current pipeline
- `src/llmssycoph/`: main package for dataset prep, sampling, probes, outputs, and runtime helpers
- `src/llmssycoph/grading/`: answer parsing, correctness grading, graded record preparation, and probe-data assembly
- `src/llmssycoph/grading/MULTIPLE_CHOICE_DEFINITIONS.md`: strict-MC terminology and metric definitions
- `pyproject.toml`: packaging metadata for the `src` layout and editable installs
- `RESULTS_FORMAT.md`: artifact layout, cache rules, and parsing guide for run outputs
- `jobs/sycophancy_bias_probe/`: SLURM job scripts for cluster runs
- `notebooks/`: downstream analysis notebooks
- `data/`: local datasets used by the experiments
- `legacy/`: older scripts from the earlier codebase, retained for reference only
- `script.py`: legacy compatibility surface retained for older workflows; not used by the main runner

## Setup

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for non-trivial runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The repository now uses a `src` layout, and `pip install -e .` keeps the `llmssycoph` package importable for tests, scripts, and notebooks.

The pipeline loads Hugging Face models and uses the sycophancy evaluation data in `data/sycophancy-eval/`. If those files are missing, the runner can fetch them from `meg-tong/sycophancy-eval`.
For direct runs, `.env` is optional. If it is missing, the code falls back to your shell environment and Hugging Face's default cache location.

Recommended `.env` keys:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: optional Hugging Face access token used when loading gated or private Hugging Face repos such as Llama-family models. If neither key is present, public Hugging Face models still work as before.
- `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE`: optional cache directory for Hugging Face model/tokenizer downloads.
- `HF_DATASETS_CACHE`: optional cache directory for Hugging Face datasets.
- `OPENAI_API_KEY` or `OPENAI_API_KEY_FOR_PROJECT`: required only when using an OpenAI-backed model such as `gpt-5.4-nano`.

Why these matter:

- gated Hugging Face models require both account access and a token at runtime
- cache paths keep model downloads off small home directories and make cluster jobs more stable
- OpenAI-backed models ignore the Hugging Face token but do require an OpenAI API key

The pipeline now reads `HF_TOKEN` directly and also aliases `HUGGINGFACE_TOKEN` to `HF_TOKEN` after loading `.env`, so existing `.env` files that already contain `HUGGINGFACE_TOKEN` continue to work.

## Quick start

Run the smoke / integrity test on the AYS-derived `aqua_mc` slice. The wrapper requests `--device auto`, so it prefers GPU when available and falls back to CPU otherwise. If `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, or `HF_HOME` is set, the wrapper normalizes those into a single Hugging Face cache location and passes it through explicitly:

```bash
bash jobs/sycophancy_bias_probe/smoke_aqua_mc_auto.sh
```

This wrapper now runs the pipeline first, then validates the produced artifacts and prints a compact health report.

Equivalent direct command:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device auto \
  --benchmark_source ays_mc_single_turn \
  --input_jsonl are_you_sure.jsonl \
  --dataset_name aqua_mc \
  --ays_mc_datasets aqua_mc \
  --mc_mode strict_mc \
  --smoke_test \
  --smoke_questions 12 \
  --override_sampling_cache \
  --probe_layer_min 1 \
  --probe_layer_max 4 \
  --max_new_tokens 256 \
  --sample_batch_size 1 \
  --run_name smoke_aqua_mc_mistral7b_auto_q12_l4
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
  --mc_mode strict_mc \
  --run_name ays_mc_truthful_aqua_run
```

Run the same strict-MC pipeline on CommonsenseQA. On first use, the loader normalizes `tau/commonsense_qa` into the same local JSONL row family under `data/sycophancy-eval/commonsense_qa.jsonl`, then the rest of the pipeline reuses the existing AYS-derived path unchanged:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --benchmark_source ays_mc_single_turn \
  --input_jsonl are_you_sure.jsonl \
  --ays_mc_datasets commonsense_qa \
  --dataset_name commonsense_qa \
  --mc_mode strict_mc \
  --run_name commonsense_qa_strict_mc_run
```

Run the same path on ARC-Challenge. On first use, the loader normalizes `allenai/ai2_arc` with config `ARC-Challenge` into `data/sycophancy-eval/arc_challenge.jsonl`. Because ARC-Challenge already ships `train`, `validation`, and `test`, the pipeline preserves those native splits instead of re-splitting the questions locally:

```bash
python run_sycophancy_bias_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --benchmark_source ays_mc_single_turn \
  --input_jsonl are_you_sure.jsonl \
  --ays_mc_datasets arc_challenge \
  --dataset_name arc_challenge \
  --mc_mode strict_mc \
  --run_name arc_challenge_strict_mc_run
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
- `--ays_mc_datasets`: comma-separated AYS source datasets to derive in `ays_mc_single_turn` mode; default is `truthful_qa_mc,aqua_mc`, and it also supports normalized HF-backed sources such as `commonsense_qa` and `arc_challenge`
- `--mc_mode`: `strict_mc` for the canonical benchmark path, or `mc_with_rationale` for the auxiliary rationale-preserving path
- strict MC prompts require `Answer: <LETTER>` and explicitly forbid non-answers such as `None`, `unknown`, or `cannot determine`
- strict MC now reads the first answer-token distribution directly over the option letters and uses one deterministic selected-choice row per prompt
- `--n_draws`: number of sampled completions per prompt for generation-based paths; strict MC forces this to `1`
- `--temperature`: generation temperature; strict MC records this as `1.0` for bookkeeping because first-token choice scoring does not use sampling temperature
- `--max_new_tokens`: generation ceiling; if omitted, the pipeline uses `256` to avoid truncating answer-bearing completions
- `--max_questions`: limit the number of question groups
- `--test_frac`: fraction of questions reserved for the held-out test split; ignored for datasets with preserved native splits such as `arc_challenge`
- `--val_frac` / `--probe_val_frac`: fraction of the non-test questions reserved for validation during probe layer selection; ignored for datasets with preserved native splits such as `arc_challenge`
- `--probe_construction`: `auto`, `sampled_completions`, or `choice_candidates`; `auto` uses choice candidates for strict MC and sampled completions otherwise
- `--probe_example_weighting`: `model_probability` or `uniform`; strict-MC choice-candidate probes default to model-probability weighting
- `--sample_batch_size`: generation batch size
- `--hf_cache_dir`: cache directory for model and tokenizer files
- `--out_dir`: root results directory
- `--run_name`: explicit name for the run directory

## Outputs

Each run writes to:

`results/sycophancy_bias_probe/<model_slug>/<run_name>/`

Main artifacts:

- `sampled_responses.csv`: one row per sampled completion, or one deterministic strict-MC selected-choice row per prompt, including both `question_id` and `prompt_id`, the raw `question`, the rendered `prompt_text`/`prompt_template`, split membership, grading result, and for MC-derived runs the preserved choice metadata (`correct_letter`, `letters`, `answer_options`, `answers_list`) plus exported strict-MC probability columns such as `P(correct)`, `P(selected)`, and `P(A)` / `P(B)` / ...
- strict MC rows also expose compliance/audit fields such as `committed_answer`, `starts_with_answer_prefix`, `strict_format_exact`, `commitment_line`, `answer_marker_count`, `multiple_answer_markers`, and generation-stop metadata (`finish_reason`, `hit_max_new_tokens`)
- `logs/sampling_integrity_summary.json`: post-sampling compliance summary, including exact-compliance / minor-deviation / failure buckets by sampling mode and template
- `final_tuples.csv`: paired neutral and biased records for the same question and draw index, including `question`, `prompt_id_x`, `prompt_id_xprime`, `prompt_x`, `prompt_with_bias`, and prompt-template provenance after dropping ambiguous samples
- `summary_by_question.csv`: question-level aggregates across repeated draws, grouped by split, with `question`, prompt ids, prompt text, `dataset`, and prompt-template provenance retained
- `probe_candidate_scores.csv`: one row per `(prompt, answer_choice)` probe evaluation example, including candidate probability, training weight, and chosen-probe score
- `probe_metadata.json`: selected layers, validation metrics, probe-construction metadata, and saved probe paths
- `reports/summary.csv` and `reports/summary.json`: flat run-level summary table with one `overall` row, one `neutral` row, and one row per bias type
- `logs/warnings.log`: warning-only report file, created only when the run emitted warnings
- `logs/warnings_summary.json`: structured warning rollup with counts by warning code and source, plus the chronological warning list
- `reports/executive_summary.md`: quick markdown overview of the run
- `logs/sampling_records.jsonl`: resumable per-sample checkpoint state used for checkpointing and cache reuse
- `logs/sampling_manifest.json`: sampling spec and checkpoint metadata
- `run_config.json`: resolved run configuration, including normalized strict-MC settings such as `n_draws = 1`, `temperature = 1.0`, and the chosen probe-construction/weighting mode
- `status.json`: run lifecycle state
- `run_summary.json`: richer nested run summary payload for programmatic inspection
- `probe_models/`: serialized sklearn probe models

`final_tuples.csv` is the main table intended for downstream analysis.

For artifact schemas and parsing guidance, see `RESULTS_FORMAT.md`.

## Current implementation notes

- Train/validation/test splitting is done at the question level, not the sample level.
- Sampled answers are parsed into short answers and labeled as `correct`, `incorrect`, or `ambiguous`.
- Ambiguous or unparseable samples are preserved in raw outputs but excluded from paired correctness metrics and probe training.
- Probe targets are correctness labels on the probe example set: sampled completions for non-strict generation paths, or explicit teacher-forced `(prompt, answer_choice)` candidates for strict MC.
- Strict-MC candidate probes are weighted by model choice probability by default; `--probe_example_weighting uniform` turns that off.
- Probe features come from the final token of the completion string used for that probe example.
- Neutral and bias-specific probes are trained separately.
- Probe layer selection is done by validation AUC on the held-out `val` split.
- After selecting the best layer, the final probe is retrained on `train + val` before scoring records.
- For strict MC, the selected-choice probe score is still written back to `sampled_responses.csv`, and the full per-choice probe table is written to `probe_candidate_scores.csv`.
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

The repository still contains older scripts under `legacy/` from the earlier hallucination-focused codebase. They are not the source of truth for the current sycophancy pipeline.
