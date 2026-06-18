# Full All-Families + Paraphrase Batch 2026-06-14

Note: for new full experiments, prefer `../full_allfamilies_paraphrase_sharded_20260616/`. This monolithic array remains useful for reference and direct reruns, but the sharded bundle is cheaper to retry and avoids mixing sampling, probe training, cross-family evaluation, and paraphrase movement in one long job.

This bundle launches the full intended experiment for the two main datasets and the two main local models:

- datasets: `commonsense_qa`, `arc_challenge`
- models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`

## What This Batch Includes

Each task runs with:

- all supported trainable non-neutral probe families from `trainable_prompt_families(include_neutral=False)`:
  - `incorrect_suggestion`
  - `incorrect_suggestion_strong`
  - `doubt_correct`
  - `doubt_correct_strong`
  - `doubt_random`
  - `doubt_random_strong`
  - `suggest_correct`
  - `suggest_correct_strong`
  - `suggest_random`
  - `random_all`
  - `suggest_random_strong`
- same-family paraphrase movement evaluation via `--paraphrase_artifact_path data/ad_hoc/paraphrase_robustness_test_stems_v1`

The family list is resolved from the Python registry at job runtime unless `BIAS_TYPES_CSV` is explicitly set, so newly added trainable probe families are included by default.

By default this bundle now uses stable run names so failed jobs can reuse compatible sampling
checkpoints from previous attempts. Set `FRESH_RUN=1` only when you explicitly want a new isolated
run directory.

## Cross-Family Evaluation

The pipeline automatically performs chosen-probe cross-family evaluation across the enabled family set.

That means when this batch enables all families, each trained probe is evaluated on:

- its own training family
- every other enabled prompt family
- neutral prompts

The paraphrase package adds same-family paraphrase movement evaluation on top of that. It is not a replacement for the prompt-family cross-evaluation matrix.

## Phase Mapping

The single runner covers the requested phases as follows:

1. strict-MC response scoring for every enabled prompt family, including prompt-level correctness / `P(correct)`
2. probe layer selection and probe training for neutral plus every enabled prompt family
3. held-out cross-family probe evaluation plus same-family paraphrase movement evaluation
4. final sampled-response tables, probe artifacts, summaries, warning reports, and executive summary

## Slurm Structure

This is submitted as one Slurm array job with 4 tasks:

- task `0`: `commonsense_qa` x `Llama-3.1-8B-Instruct`
- task `1`: `commonsense_qa` x `Qwen2.5-7B-Instruct`
- task `2`: `arc_challenge` x `Llama-3.1-8B-Instruct`
- task `3`: `arc_challenge` x `Qwen2.5-7B-Instruct`

This is the cleanest “one Slurm submission” version while keeping failures isolated per model/dataset run.

## Resources

The sbatch file declares:

- `gpu,seas_gpu,gpu_h200`
- `1` GPU
- `2` CPUs
- `100G` CPU RAM
- `24:00:00` wall time

The submit wrapper overrides wall time to `48:00:00` by default because the full
CommonsenseQA all-family probe/eval phase can exceed 24 hours. You can override resources
without editing the sbatch file:

```bash
SBATCH_TIME=72:00:00 SBATCH_MEM=120G bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh
```

The run is substantially heavier than the default-family batch because it expands both the number of probe families and the cross-family evaluation matrix.

## Logs

Slurm logs go to:

- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614/`

Emails are sent on:

- `END`
- `FAIL`

## Submit

One command:

```bash
bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh
```

Useful rerun/debug variants:

```bash
# Print the final sbatch command without submitting.
DRY_RUN=1 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh

# Rerun only task 1, CommonsenseQA x Qwen2.5-7B.
SBATCH_ARRAY=1 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh

# Force a completely isolated fresh run.
FRESH_RUN=1 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh

# Clear a stale lock for the stable run name after confirming no matching job is active.
ALLOW_STALE_LOCK_CLEANUP=1 SBATCH_ARRAY=0 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh

# Small wiring test that still exercises all prompt families and paraphrase lookup.
MAX_QUESTIONS=12 PROBE_SELECTION_MAX_SAMPLES=200 PROBE_TRAIN_MAX_SAMPLES=400 SBATCH_ARRAY=0 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh
```
