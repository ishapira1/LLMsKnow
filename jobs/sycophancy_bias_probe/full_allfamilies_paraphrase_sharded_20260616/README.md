# Full All-Families Paraphrase Sharded 20260616

Recommended full experiment path for the ARC/CSQA x Llama/Qwen strict-MC runs.

This bundle splits the old monolithic full job into two Slurm stages:

1. `sampling_array.sbatch`: samples and scores every prompt family once for each dataset/model.
2. `probe_family_array.sbatch`: trains/evaluates one probe family per array task, reusing the sampling cache from stage 1.

The default probe-family list comes from:

```bash
python -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=True)))'
```

With the current registry this is 12 probe families, so the default probe array is 48 tasks: 4 dataset/model groups x 12 probe families. This includes `neutral`, `doubt_random`, `doubt_random_strong`, and `random_all`.

## Submit

```bash
bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

The submitter launches sampling first, then submits the probe-family array with `afterok` dependency.

## Logs

The submit wrapper creates:

- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260616/submit/`
- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260616/slurm/sampling/`
- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260616/slurm/probes/`
- `jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260616/by_task/`

Slurm stdout/stderr uses informative stage names:

```text
submit/submit_<timestamp>_pid_<pid>.log
slurm/sampling/syco_allfam_sample_20260616.<array_job>_<task>.out
slurm/sampling/syco_allfam_sample_20260616.<array_job>_<task>.err
slurm/probes/syco_allfam_probe_20260616.<array_job>_<task>.out
slurm/probes/syco_allfam_probe_20260616.<array_job>_<task>.err
```

Each task also tees to a canonical labeled structured log:

```text
by_task/<dataset_model>/sampling/job_<job_id>/task_<task>.out
by_task/<dataset_model>/sampling/job_<job_id>/task_<task>.err
by_task/<dataset_model>/probe_<family>/job_<job_id>/task_<task>.out
by_task/<dataset_model>/probe_<family>/job_<job_id>/task_<task>.err
```

Use `by_task/` first when debugging a specific dataset/model/probe shard. The task logs print the resolved run name, run directory, command, Slurm IDs, hostname, start/end times, `nvidia-smi`, and a best-effort `sstat` snapshot.

## Memory Defaults

The sharded scripts keep `SAMPLE_BATCH_SIZE=1` by default and set conservative runtime defaults:

- `PYTHONUNBUFFERED=1`
- `TOKENIZERS_PARALLELISM=false`
- `MALLOC_ARENA_MAX=2`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

These do not change the experiment semantics. They make logs stream promptly, avoid tokenizer thread oversubscription, reduce glibc arena growth, and reduce CUDA allocator fragmentation risk.

## Dry Run And Filters

```bash
DRY_RUN=1 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

Rerun a single probe family:

```bash
PROBE_FAMILY_FILTER=suggest_random bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

Rerun a single dataset/model group:

```bash
TASK_FILTER=commonsense_qa_qwen25_7b bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

Supported task labels:

- `commonsense_qa_llama31_8b`
- `commonsense_qa_qwen25_7b`
- `arc_challenge_llama31_8b`
- `arc_challenge_qwen25_7b`

If sampling already completed and only probes should be submitted:

```bash
SUBMIT_PROBES_ONLY=1 SAMPLING_JOB_ID=<completed_sampling_job_id> bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

Stale lock cleanup is intentionally explicit:

```bash
ALLOW_STALE_LOCK_CLEANUP=1 PROBE_FAMILY_FILTER=neutral TASK_FILTER=arc_challenge_qwen25_7b bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/submit_full_allfamilies_paraphrase_sharded_20260616.sh
```

## Run Names

Sampling:

```text
<dataset>_<model>_allfamilies_sampling_20260616
```

Probe shards:

```text
<dataset>_<model>_allfamilies_probe_<family>_20260616
```

No `--fresh_run` is passed by default. Probe shards pass the full `--bias_types` list and a single `--probe_families` value, so they reuse the sampling-only cache while writing to isolated run directories.
