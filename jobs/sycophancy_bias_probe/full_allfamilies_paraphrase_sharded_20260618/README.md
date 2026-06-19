# Full All-Families Paraphrase Sharded 20260618

Recommended full experiment bundle for the current strict-MC local-model runs on the two main datasets:

- `commonsense_qa`
- `arc_challenge`

with the two current main local models:

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

This bundle follows the learned safe pattern:

1. run one sampling-only array over the 4 `dataset x model` combinations
2. run a dependent probe-family array with one probe family per task
3. validate the produced artifacts at the end of each task
4. keep scheduler logs and human-browseable task logs in a structured tree

Same-family paraphrase movement evaluation is enabled by default through:

`data/ad_hoc/paraphrase_robustness_test_stems_v1`

The new run-level external paraphrase evaluation is also enabled by default, but only in the
sampling-only stage so it runs once per `dataset x model` run rather than once per probe shard.

## Files

- `sampling_array.sbatch`: sampling-only array over the 4 dataset/model combinations
- `probe_family_array.sbatch`: one probe family per array task, reusing the sampling cache
- `submit_full_allfamilies_paraphrase_sharded_20260618.sh`: submits sampling first, then probes with `afterok`
- `status_full_allfamilies_paraphrase_sharded_20260618.sh`: summarizes Slurm state plus run-directory completion

## Submit

```bash
bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/submit_full_allfamilies_paraphrase_sharded_20260618.sh
```

Dry run:

```bash
DRY_RUN=1 bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/submit_full_allfamilies_paraphrase_sharded_20260618.sh
```

By default on the Harvard cluster, this bundle keeps heavy outputs off the home quota filesystem:

- results: `/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/sycophancy_bias_probe`
- submit, Slurm, and task logs: `/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618`
- HF, Triton, W&B, Torch, Matplotlib, XDG, and temp caches: under the resolved `SYCOPHANCY_STORAGE_ROOT`

Override `SYCOPHANCY_STORAGE_ROOT`, `OUT_DIR`, or `LOG_ROOT` only if you are pointing to non-home storage. The wrappers reject result/log/cache paths under `/home` or `/n/home`.

## Status

```bash
bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/status_full_allfamilies_paraphrase_sharded_20260618.sh
```

That status helper reads the latest submission metadata by default and reports:

- sampling job and probe job IDs
- current `squeue` / `sacct` view when available
- expected vs completed sampling runs
- expected vs completed probe-family runs
- first incomplete or failed run path when one exists

## Logs

By default, all logs live under:

`/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/`

Important subtrees:

- `submit/`
- `slurm/sampling/`
- `slurm/probes/`
- `by_task/<dataset_model>/sampling/job_<job_id>/task_<task>.out`
- `by_task/<dataset_model>/probe_<family>/job_<job_id>/task_<task>.out`

Each task log prints:

- task label
- dataset and model
- probe family when applicable
- run name and run directory
- exact command
- Slurm IDs
- hostname and working directory
- start/end timestamps
- elapsed seconds
- `nvidia-smi` snapshots
- best-effort `sstat` resource snapshot
- post-run artifact verification summary

The pipeline already emits `tqdm` progress bars for sampling, probe selection, scoring, and movement evaluation, and those are preserved in the structured task logs.

## Task Labels

- `commonsense_qa_llama31_8b`
- `commonsense_qa_qwen25_7b`
- `arc_challenge_llama31_8b`
- `arc_challenge_qwen25_7b`

## Probe Families

By default the submitter resolves the current trainable family registry with:

```bash
python -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=True)))'
```

With the current registry this yields 12 probe families, so the default probe array is `4 x 12 = 48` tasks.

## Useful Filters

Single dataset/model group:

```bash
TASK_FILTER=arc_challenge_qwen25_7b bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/submit_full_allfamilies_paraphrase_sharded_20260618.sh
```

Single probe family:

```bash
PROBE_FAMILY_FILTER=neutral bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/submit_full_allfamilies_paraphrase_sharded_20260618.sh
```

Probe-only resubmission after sampling already finished:

```bash
SUBMIT_PROBES_ONLY=1 SAMPLING_JOB_ID=<sampling_job_id> bash jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260618/submit_full_allfamilies_paraphrase_sharded_20260618.sh
```

## Safety Defaults

- `#SBATCH --mail-type=END,FAIL`
- `#SBATCH --mail-user=itaishapira@g.harvard.edu`
- `SAMPLE_BATCH_SIZE=1`
- `PYTHONUNBUFFERED=1`
- `TOKENIZERS_PARALLELISM=false`
- `MALLOC_ARENA_MAX=2`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- no stale lock deletion unless `ALLOW_STALE_LOCK_CLEANUP=1`

## Post-Run Verification

The Slurm tasks do a final artifact check after the pipeline exits successfully.

Sampling tasks verify at least:

- `meta/status.json`
- `meta/run_manifest.json`
- `sampling/raw/sampling_manifest.json`
- `sampling/flat/sampled_responses.csv`

Probe-family tasks additionally verify:

- `query/chosen_probe_registry.csv`
- `query/chosen_probe_metrics.csv`
- `query/chosen_probe_cross_family_metrics.csv`
- `query/chosen_probe_movement_summary.csv`
- `query/chosen_probe_movement_items.jsonl`
- `query/paraphrase_coverage.csv`

This does not replace deeper analysis, but it catches the common failure mode where the Python process exits cleanly while the expected result tree is incomplete.
