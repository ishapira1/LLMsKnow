# Bonham paper-ready sparse-pruning campaign

This is an isolated, fail-closed implementation of `pruning_bonham_20260918` for Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-4-12B-Instruct. It does not modify or import artifacts from completed pruning campaigns, except that it deliberately reuses the repository's established benchmark task builders and pinned EvalPlus sandbox contract.

The frozen protocol is [`configs/experiments/pruning_bonham_20260918.json`](../../../configs/experiments/pruning_bonham_20260918.json). Every target mask contains exactly 1,000 coordinates and records `p=0.00005`. N1 and N2 use byte-identical pruning manifests and one shared immutable pruning-score cache. Their preservation scores differ: N1 uses 512 general examples; N2 uses those same 512 bytes followed by 512 source-responsive examples.

## Safety and reproducibility contracts

- Source files, selected rows, rendered prompts, manifests, score tensors, masks, evaluation cells, and reports are immutable after publication.
- `COMPLETE` or `COMPLETE.json` receipts authenticate every published stage.
- Preservation scoring accumulates `abs(-weight * gradient)` separately for each example before taking the mean.
- Scoring is restricted to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` matrices.
- Selection protects the top `p` fraction within each matrix, computes pruning percentiles among that matrix's remaining candidates, pools percentiles, and uses SHA-256 coordinate tie-breaking.
- Bare-user rows are rejected from N2's preservation bank.
- Steering fit/development questions and all preservation questions are disjoint from mask and final-evaluation questions.
- Rationale-framed pushback is evaluated as a separate multi-turn stress test using the four fixed prompts in [`reasoning_backed_pushback_templates.json`](reasoning_backed_pushback_templates.json), balanced exactly within each 500-question factual dataset. Its registry is authenticated when evaluation manifests are frozen. It is excluded from the primary four-category macro-average because its justifications are generic rather than question-grounded.
- No weight coordinate is compared across model architectures.
- Stale `.run.lock` cleanup is never automatic. `ALLOW_STALE_LOCK_CLEANUP` must remain `0`.

## Local verification

From the repository root:

```bash
./.venv/bin/python jobs/sycophancy_pruning/pruning_bonham_20260918/test_bundle.py -v
bash -n jobs/sycophancy_pruning/pruning_bonham_20260918/common.sh
bash -n jobs/sycophancy_pruning/pruning_bonham_20260918/cpu_stage.sbatch
bash -n jobs/sycophancy_pruning/pruning_bonham_20260918/gpu_array.sbatch
bash -n jobs/sycophancy_pruning/pruning_bonham_20260918/submit.sh
DRY_RUN=1 bash jobs/sycophancy_pruning/pruning_bonham_20260918/submit.sh
```

The dry run is the full submitter preflight. It prints every `sbatch` command and dependency without submitting jobs.

## Full cluster launch

```bash
DRY_RUN=0 bash jobs/sycophancy_pruning/pruning_bonham_20260918/submit.sh
```

If a root job has already been independently validated, a recovery submission can reuse its
Slurm ID without duplicating completed source work. The submitter accepts only numeric job IDs
whose accounting state is still usable:

```bash
BONHAM_REUSE_CAPABILITY_SOURCES_JOB_ID=<job_id> \
BONHAM_REUSE_SOURCE_FREEZE_JOB_ID=<job_id> \
DRY_RUN=0 bash jobs/sycophancy_pruning/pruning_bonham_20260918/submit.sh
```

The source freeze is shared. After it completes, the three model DAGs run concurrently. Attribution is sharded by model and score role. Evaluation is sharded by model, state, suite, and question shard. There is no monolithic all-model GPU job.

The screen/evaluation arrays use conservative fixed ceilings because downstream shard counts do not exist when the initial DAG is submitted. A task whose immutable shard was not materialized exits successfully with `skip_reason=shard_not_materialized`. The ceilings cover the protocol's maximum possible shard counts:

- neutral screening: 80 shards per model;
- N1 behavioral screening: 320 shards per model;
- reliable-source screening: 48 shards per model;
- generalization: 60 shards per state/model;
- useful assertions plus native-tool transfer: 120 shards per state/model;
- capabilities: 80 shards per state/model.

## Restartable stage commands

All commands below are idempotent for already-complete artifacts and fail rather than overwrite a changed or partial final artifact. The Slurm wrappers are preferred because they create the canonical logs and resource snapshots.

```bash
# Shared source freeze and preflight
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/prepare_capability_sources.py \
  --output-root "$CAPABILITY_SOURCE_ROOT" --hf-cache "$HF_CACHE_DIR"
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py prepare-static \
  --result-root "$RESULT_ROOT" --suite-source-bindings "$SUITE_SOURCE_BINDINGS"

# One neutral or biased screen shard
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py run-screen \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --hf-cache "$HF_CACHE_DIR" \
  --stage neutral_screen --task-shard "$RESULT_ROOT/inputs/neutral_screen_shards/shard_0000.jsonl" \
  --shard-index 0 --output "$RESULT_ROOT/neutral_screen/llama31_8b/shard_0000"

# Allocation, attribution, masks, random controls, and states
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py allocate-manifests --result-root "$RESULT_ROOT"
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py score-component \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --score-id n1_seed5_prune --hf-cache "$HF_CACHE_DIR"
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py build-masks \
  --result-root "$RESULT_ROOT" --model-key llama31_8b
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py build-random-mask \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --hf-cache "$HF_CACHE_DIR"
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py build-mask-states \
  --result-root "$RESULT_ROOT" --model-key llama31_8b

# MeanDiff fit and development
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/steering.py prepare \
  --result-root "$RESULT_ROOT" --model-key llama31_8b
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/steering.py extract \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --hf-cache "$HF_CACHE_DIR"
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/steering.py develop \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --hf-cache "$HF_CACHE_DIR"

# Freeze and execute one evaluation cell
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/evaluations.py prepare \
  --result-root "$RESULT_ROOT" --suite-source-bindings "$SUITE_SOURCE_BINDINGS" \
  --external-utility-root "$CAPABILITY_SOURCE_ROOT"
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/evaluations.py run-shard \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --state-id n1_mechanism \
  --family generalization --shard 0 --hf-cache "$HF_CACHE_DIR"

# Weight analysis, reporting, and audit
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/weight_analysis.py analyze-model \
  --result-root "$RESULT_ROOT" --model-key llama31_8b
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/reporting.py --result-root "$RESULT_ROOT"
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/audit.py --result-root "$RESULT_ROOT"
```

## Artifact and log layout

The default result root is:

```text
/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/pruning_bonham_20260918/
```

Major subtrees are `inputs/`, `manifests/`, `scores/`, `masks/`, `states/`, `steering/`, `evaluations/`, `evalplus/`, `weight_analysis/`, `reports/`, and `audit/`.

Logs follow the repository convention under `jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/`:

```text
submit/
slurm/<stage>/
by_task/<model>/<stage>/job_<job_id>/task_<array_task>.out
by_task/<model>/<stage>/job_<job_id>/task_<array_task>.err
```

Each task records its label, model, stage, command, Slurm IDs, host, working directory, start/end times, exit status, elapsed seconds, and available `nvidia-smi`/`sstat` snapshots.
