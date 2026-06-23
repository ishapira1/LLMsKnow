# Anti-Sycophancy Request Sharded Bundle, 2026-06-23

This bundle runs the external-only anti-sycophancy request experiment over:

- `commonsense_qa` and `arc_challenge`
- `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`
- request strengths `weak` and `strong`

The sampling stage creates 8 request runs:

```text
4 dataset/model combinations x 2 request strengths
```

Each sampling task runs `run_sycophancy_bias_probe.py --sampling_only` with all current trainable non-neutral prompt families and `--anti_sycophancy_request <weak|strong>`.

The analysis stage creates 8 dependent comparison tasks. Each task compares the request run against the no-request baseline sampling run and scores the saved baseline `probe_bias_random_all` artifact only. No probes are retrained.

The final summary stage is one dependent job. It aggregates all `behavior_summary.csv` files and emails the main neutral-correct-conditioned mitigation statistics.

## Defaults

- Bundle name: `anti_sycophancy_request_sharded_20260623`
- Baseline tag: `20260618`
- Request tag: `20260623`
- Baseline sampling run name: `<run_slug>_allfamilies_sampling_20260618`
- Baseline random-all probe run name: `<run_slug>_allfamilies_probe_random_all_20260618`
- Request sampling run name: `<run_slug>_antisyc_<weak|strong>_sampling_20260623`
- Comparison output: `$OUT_DIR/_comparisons/anti_sycophancy_request_20260623/<dataset_model>/<weak|strong>/`
- Slurm lifecycle email: `BEGIN,END,FAIL`
- Custom summary email recipient: `itaishapira@g.harvard.edu`

## Submit

Dry-run first:

```bash
DRY_RUN=1 bash jobs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/submit_anti_sycophancy_request_sharded_20260623.sh
```

Submit:

```bash
bash jobs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/submit_anti_sycophancy_request_sharded_20260623.sh
```

Useful overrides:

- `TASK_FILTER=commonsense_qa_llama31_8b`
- `REQUEST_FILTER=weak` or `REQUEST_FILTER=strong`
- `BASELINE_TAG=20260618`
- `REQUEST_TAG=20260623`
- `BASELINE_SAMPLING_RUN_DIR_<TASK_LABEL>=/path/to/run`
- `BASELINE_RANDOM_ALL_PROBE_RUN_DIR_<TASK_LABEL>=/path/to/run`
- `MAX_PROBE_PAIRS=128` for smoke-test analysis
- `DEVICE=auto`
- `DEVICE_MAP_AUTO=1`
- `SUMMARY_EMAIL_TO=you@example.edu`
- `SUMMARY_EMAIL_SUBJECT="Anti-sycophancy request results"`
- `SEND_SUMMARY_EMAIL=0` to write summary files without sending mail

Per-task baseline override variable names use uppercase task labels:

- `BASELINE_SAMPLING_RUN_DIR_COMMONSENSE_QA_LLAMA31_8B`
- `BASELINE_RANDOM_ALL_PROBE_RUN_DIR_COMMONSENSE_QA_LLAMA31_8B`
- `BASELINE_SAMPLING_RUN_DIR_COMMONSENSE_QA_QWEN25_7B`
- `BASELINE_RANDOM_ALL_PROBE_RUN_DIR_COMMONSENSE_QA_QWEN25_7B`
- `BASELINE_SAMPLING_RUN_DIR_ARC_CHALLENGE_LLAMA31_8B`
- `BASELINE_RANDOM_ALL_PROBE_RUN_DIR_ARC_CHALLENGE_LLAMA31_8B`
- `BASELINE_SAMPLING_RUN_DIR_ARC_CHALLENGE_QWEN25_7B`
- `BASELINE_RANDOM_ALL_PROBE_RUN_DIR_ARC_CHALLENGE_QWEN25_7B`

## Logs

Logs are rooted at:

```text
jobs/sycophancy_bias_probe/logs/anti_sycophancy_request_sharded_20260623/
```

or the cluster storage equivalent from `storage_common.sh`.

Layout:

- `submit/`: submitter logs, task matrices, latest submission env
- `slurm/sampling/`: raw Slurm stdout/stderr for sampling
- `slurm/analysis/`: raw Slurm stdout/stderr for comparison
- `slurm/summary/`: raw Slurm stdout/stderr for the final summary-email job
- `by_task/<dataset_model>/<weak|strong>/sampling/job_<job_id>/task_<array_task>.out`
- `by_task/<dataset_model>/<weak|strong>/analysis/job_<job_id>/task_<array_task>.out`
- `by_task/summary/job_<job_id>/task_summary.out`

Matching `.err` files are written beside `.out` files.

## Email Summary Contents

The final custom email reports:

- `baseline_drop = 1 - accuracy(biased family, no request)`
- `request_drop = 1 - accuracy(biased family, weak/strong request)`
- `mitigation = baseline_drop - request_drop`

All rows are conditioned on the baseline no-request neutral prompt being usable and correct. The email includes topline averages over non-neutral prompt families and a separate `random_all` row for each dataset/model/request strength.
