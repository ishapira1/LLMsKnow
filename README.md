# Hidden Knowledge in Sycophancy

This repository contains the code for measuring whether a model's sycophantic behavior reflects genuine uncertainty or a mismatch between what its hidden states support and what it ultimately says.

The active code path samples or scores responses to neutral and bias-injected prompts, labels answer correctness, trains linear probes on hidden states, and compares internal evidence against observed behavior.

## Current main datasets

For the current project, the main datasets are:

- `arc_challenge`
- `commonsense_qa`

The repo still supports older/default slices such as `aqua_mc` and `truthful_qa_mc`, but the main full-run analysis and cluster workflows are centered on `arc_challenge` and `commonsense_qa`.

Project convention:
- In future project discussions, analyses, and AI-assisted conversations, treat `arc_challenge` and `commonsense_qa` as the default main datasets unless a different dataset is explicitly requested.
- Treat `aqua_mc` as a lightweight smoke/sanity dataset, not the default scientific focus of the project.

## What the pipeline does

For each question, the pipeline builds:

- a neutral prompt `x`
- one or more non-neutral prompt-family variants `x'`

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

## Prompt-family convention

The canonical internal term is `prompt family`.

- `neutral` is the baseline prompt family.
- The current user-selectable non-neutral prompt families are:
  - `incorrect_suggestion`
  - `incorrect_suggestion_strong`
  - `doubt_correct`
  - `doubt_correct_strong`
  - `doubt_random`
  - `doubt_random_strong`
  - `random_all`
  - `suggest_correct`
  - `suggest_correct_strong`
  - `suggest_random`
  - `suggest_random_strong`
- Serialized artifacts still retain legacy field names such as `template_type` and derived reporting names such as `bias_type` for compatibility.
- In older notes, notebooks, or saved tables, you may still see phrases like "bias template" or "prompt template". In the current code, those should usually be read as "prompt family" unless the text is explicitly referring to the legacy serialized field `prompt_template`.

`suggest_random` is a suggestion-style control family for MC-derived runs. It chooses exactly one answer option uniformly from the available choices, including the correct option, and renders the option text rather than the option letter. The chosen suggestion is deterministic within a run given the run seed and question identity.

`doubt_random` mirrors that same seeded random-option selection, but negates the chosen option instead of endorsing it. It is distinct from `doubt_correct`, which specifically negates the true answer.

`random_all` is a seeded mixture family for MC-derived runs. For each question, it deterministically chooses one random answer option and then deterministically chooses whether to render that option with the weak `suggest_random` wording or the weak `doubt_random` wording. This lets the corresponding probe train on a fixed within-run mixture of both framings.

`incorrect_suggestion_strong` reuses the same user-endorsed incorrect answer target as `incorrect_suggestion`, but changes the wording to a much stronger, more authoritative claim.

The other core strong prompt families mirror that same idea:
- `doubt_correct_strong` is the strong-confidence version of `doubt_correct`
- `doubt_random_strong` is the strong-confidence version of `doubt_random`
- `suggest_correct_strong` is the strong-confidence version of `suggest_correct`
- `suggest_random_strong` is the strong-confidence version of `suggest_random`

## Probe interpretation convention

Be careful when interpreting probe scores across prompt conditions.

- The pipeline trains a neutral probe and separate prompt-family probes.
- The main motivation for the non-neutral prompt-family probes is as a sanity check / auxiliary diagnostic.
- In our typical scientific interpretation, `s` should mean the neutral probe.
- That means `s(x, a)` is the neutral probe evaluated on the neutral prompt, and `s(x', a)` is that same neutral probe evaluated on the biased prompt.
- If an analysis instead compares a neutral probe on `x` to a non-neutral prompt-family probe on `x'`, it should say so explicitly, because that is not the default intended interpretation.
- In particular, the saved prompt-level probe table may contain matched prompt-family probe scores by prompt family, so downstream notebooks should be explicit about whether they are using matched probes or rescoring `x'` with the neutral probe.

## Repository layout

- `run_sycophancy_bias_probe.py`: thin public wrapper for the current pipeline
- `src/llmssycoph/`: main package for dataset prep, sampling, probes, outputs, and runtime helpers
- `src/llmssycoph/data/prompt_families.py`: canonical registry for prompt-family rendering, detection, ordering, and probe-name mapping
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

For the current project, use `commonsense_qa` and `arc_challenge` for substantive runs. The `aqua_mc` examples below are mainly for lightweight smoke tests and debugging.

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

If you want to force a clearly isolated rerun with no sampling-cache reuse, add:

```bash
python run_sycophancy_bias_probe.py \
  ...your usual args... \
  --fresh_run
```

This disables sampling-cache reuse and creates a fresh, clearly labeled run directory. If you also pass
`--run_name`, the runner appends a fresh-run suffix instead of reusing the old directory.

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

Run the same strict-MC pipeline on CommonsenseQA. This is one of the two main datasets for the current project. On first use, the loader normalizes `tau/commonsense_qa` into the same local JSONL row family under `data/sycophancy-eval/commonsense_qa.jsonl`, then the rest of the pipeline reuses the existing AYS-derived path unchanged:

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

Run the same path on ARC-Challenge. This is one of the two main datasets for the current project. On first use, the loader normalizes `allenai/ai2_arc` with config `ARC-Challenge` into `data/sycophancy-eval/arc_challenge.jsonl`. Because ARC-Challenge already ships `train`, `validation`, and `test`, the pipeline preserves those native splits instead of re-splitting the questions locally:

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
- `--bias_types`: comma-separated subset of `incorrect_suggestion`, `incorrect_suggestion_strong`, `doubt_correct`, `doubt_correct_strong`, `doubt_random`, `doubt_random_strong`, `random_all`, `suggest_correct`, `suggest_correct_strong`, `suggest_random`, `suggest_random_strong`. By default the CLI uses every trainable non-neutral prompt family, so `random_all` is included automatically in the standard training and cross-family evaluation runs.
- `--dataset_name` / `--dataset_type`: source dataset from `base.dataset` to keep, or `all` to use every dataset. For the current project, this will usually be `commonsense_qa` or `arc_challenge`.
- `--ays_mc_datasets`: comma-separated AYS source datasets to derive in `ays_mc_single_turn` mode; default is `truthful_qa_mc,aqua_mc`, and it also supports normalized HF-backed sources such as `commonsense_qa` and `arc_challenge`. For the current project, the main settings are `commonsense_qa` and `arc_challenge`.
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
- `--fresh_run`: disable sampling-cache reuse and force a clearly isolated run directory
- `--run_name`: explicit name for the run directory

## Outputs

Each run writes to:

`results/sycophancy_bias_probe/<model_slug>/<dataset_dir>/<run_name>/`

Core pipeline artifacts:

- `logs/run.log`: human-readable runtime log
- `logs/sampling_records.jsonl`: canonical raw sampling store
- `logs/sampling_manifest.json`: sampling spec and checkpoint metadata
- `logs/sampling_integrity_summary.json`: post-sampling integrity summary
- `logs/warnings.log` and `logs/warnings_summary.json`: optional warning artifacts, present only when warnings were emitted
- `sampling/sampled_responses.csv`: flat per-record analysis table
- `reports/summary.json` and `reports/summary.csv`: flat run-level summary rows
- `reports/executive_summary.md`: quick markdown overview
- `reports/confusion_matrix_predicted_letter_x_true_letter.csv`: optional MC confusion-matrix export
- `probes/probe_scores_by_prompt.csv`: prompt-level probe table, written even for `--sampling_only`
- `run_config.json`: resolved config plus artifact-path metadata
- `run_summary.json`: richer nested summary payload
- `run_summary.json.runtime_timing`: structured top-level stage timing plus nested probe substage timing
- `status.json`: run lifecycle state

Optional probe-training artifacts:

- `probes/all_probes/`: all trained layer candidates and manifests
- `probes/chosen_probe/`: final selected probes and manifests

Optional post-hoc derived artifacts:

- `analysis/`: notebook status, notebooks, plots, and tables created later by analysis scripts
- `sampling_backfills/` and `probes/backfills/`: later derived backfill/rescoring outputs

The base pipeline does not write post-hoc analysis or backfill artifacts by default.

For artifact schemas and parsing guidance, see `RESULTS_FORMAT.md`.

## Current implementation notes

- Train/validation/test splitting is done at the question level, not the sample level.
- Sampled answers are parsed into short answers and labeled as `correct`, `incorrect`, or `ambiguous`.
- Ambiguous or unparseable samples are preserved in raw outputs but excluded from paired correctness metrics and probe training.
- Probe targets are correctness labels on the probe example set: sampled completions for non-strict generation paths, or explicit teacher-forced `(prompt, answer_choice)` candidates for strict MC.
- Strict-MC candidate probes are weighted by model choice probability by default; `--probe_example_weighting uniform` turns that off.
- Probe features come from the final token of the completion string used for that probe example.
- Neutral and bias-specific probes are trained separately.
- Bias-specific probes are mainly a sanity check and auxiliary diagnostic; unless an analysis explicitly says otherwise, the intended interpretation of `s` is the neutral probe, including when evaluating biased prompts `x'`.
- Probe layer selection is done by validation AUC on the held-out `val` split.
- After selecting the best layer, the final probe is retrained on `train + val` before scoring records.
- The probe-heavy portion of the run is timed as explicit substages: record-set assembly, layer selection, retraining/in-family scoring, cross-family evaluation, and artifact persistence.
- For strict MC, the selected-choice probe score is still written back to `sampling/sampled_responses.csv`, and the exported core probe table is `probes/probe_scores_by_prompt.csv`.
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
