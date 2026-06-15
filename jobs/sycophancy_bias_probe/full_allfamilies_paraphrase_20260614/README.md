# Full All-Families + Paraphrase Batch 2026-06-14

This bundle launches the full intended experiment for the two main datasets and the two main local models:

- datasets: `commonsense_qa`, `arc_challenge`
- models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`

## What This Batch Includes

Each task runs with:

- all supported trainable bias families:
  - `incorrect_suggestion`
  - `incorrect_suggestion_strong`
  - `doubt_correct`
  - `doubt_correct_strong`
  - `suggest_correct`
  - `suggest_correct_strong`
  - `suggest_random`
  - `suggest_random_strong`
- `--paraphrase_artifact_path data/ad_hoc/paraphrase_robustness_test_stems_v1`
- `--fresh_run`

## Cross-Family Evaluation

The pipeline automatically performs chosen-probe cross-family evaluation across the enabled family set.

That means when this batch enables all families, each trained probe is evaluated on:

- its own training family
- every other enabled prompt family
- neutral prompts

The paraphrase package adds same-family paraphrase movement evaluation on top of that. It is not a replacement for the prompt-family cross-evaluation matrix.

## Slurm Structure

This is submitted as one Slurm array job with 4 tasks:

- task `0`: `commonsense_qa` x `Llama-3.1-8B-Instruct`
- task `1`: `commonsense_qa` x `Qwen2.5-7B-Instruct`
- task `2`: `arc_challenge` x `Llama-3.1-8B-Instruct`
- task `3`: `arc_challenge` x `Qwen2.5-7B-Instruct`

This is the cleanest “one Slurm submission” version while keeping failures isolated per model/dataset run.

## Resources

The array uses:

- `gpu,seas_gpu,gpu_h200`
- `1` GPU
- `2` CPUs
- `100G` CPU RAM
- `24:00:00` wall time

The run is substantially heavier than the default-family batch because it expands both the number of probe families and the cross-family evaluation matrix.

## Logs

Slurm logs go to:

- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614/`

Emails are sent on:

- `BEGIN`
- `END`
- `FAIL`

## Submit

One command:

```bash
bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/submit_full_allfamilies_paraphrase_20260614.sh
```
