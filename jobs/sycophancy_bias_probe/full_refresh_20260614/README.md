# Full Refresh Batch 2026-06-14

This bundle is the fresh full-experiment Slurm batch for the current canonical local-model runs.

## Split Strategy

The jobs are split by `dataset x model`:

- `commonsense_qa` x `Llama-3.1-8B-Instruct`
- `commonsense_qa` x `Qwen2.5-7B-Instruct`
- `arc_challenge` x `Llama-3.1-8B-Instruct`
- `arc_challenge` x `Qwen2.5-7B-Instruct`

That split keeps each run isolated at the artifact level and makes it easy to resubmit a single model or dataset without touching the others.

## Fresh-Run Behavior

Each job passes:

- a readable dated base `--run_name`
- `--fresh_run`

So the saved run directory stays human-readable while still forcing a fresh isolated run instead of resuming a previous one.

## Logs

Slurm stdout/stderr is saved outside the results tree:

- `jobs/sycophancy_bias_probe/logs/full_refresh_20260614/commonsense_qa/`
- `jobs/sycophancy_bias_probe/logs/full_refresh_20260614/arc_challenge/`

## Email Notifications

Every job in this bundle sends Slurm email notifications to `itaishapira@g.harvard.edu` on:

- `BEGIN`
- `END`
- `FAIL`

## Submission

Submit everything:

```bash
bash jobs/sycophancy_bias_probe/full_refresh_20260614/submit_full_refresh_20260614.sh
```

Submit by dataset:

```bash
bash jobs/sycophancy_bias_probe/full_refresh_20260614/submit_commonsense_qa_full_refresh_20260614.sh
bash jobs/sycophancy_bias_probe/full_refresh_20260614/submit_arc_challenge_full_refresh_20260614.sh
```
