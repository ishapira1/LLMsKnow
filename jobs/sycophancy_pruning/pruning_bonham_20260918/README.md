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
- The main source-attribution sweep assigns exactly one of the twelve frozen credible-source templates or the native structured tool to each eligible question. Assignment is deterministic and balanced within model, dataset, and frozen initially-correct/initially-incorrect cohort. The assigned form is crossed with both applicable claim types and both turn formats, and every source result is paired with the exact same frozen bare-user question, proposition, target, truth status, and turn format.
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
- matched source-attribution sweep: 60 shards per state/model;
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
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/evaluations.py \
  prepare-source-attribution --result-root "$RESULT_ROOT"
$PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/evaluations.py run-shard \
  --result-root "$RESULT_ROOT" --model-key llama31_8b --state-id n1_mechanism \
  --family generalization --shard 0 --hf-cache "$HF_CACHE_DIR"
```

### Exact-quota Gemma recovery

If the preferred all-model construction pool and the frozen ARC-only
model-specific extension leave Gemma short in an exact N1 cell, run
`prepare-model-n1-supplement`. It freezes an append-only screen over unused,
neutral-correct CommonsenseQA construction questions for the single deficient
multi-turn incorrect-suggestion template. It changes neither the behavior
qualification criterion nor any allocation quota: Gemma must still contain 512
distinct questions and exactly 16 rows in every dataset × turn × bias ×
construction-template cell. The exact supplement is therefore preferred to the
separately guarded balance-amendment flag.

```bash
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py \
  prepare-model-n1-supplement --result-root "$RESULT_ROOT" \
  --model-key gemma4_12b --shard-size 200
```

`accelerate_gemma_exact.sh` monitors the supplement screen, promotes a still
pending regular-queue job into the first fully free `gpu_test` allocation,
runs the unchanged exact allocator, and then drives Gemma attribution, masks,
steering, and all evaluation families. Running or completed regular-queue work
is preserved rather than duplicated.

The exact 32-cell allocator remains the default and fails closed. The documented
Gemma-only balanced-marginal fallback is present as an explicit, inert opt-in so
that it cannot alter Llama or Qwen artifacts accidentally. It may be invoked only
after protocol approval:

```bash
$CPU_PYTHON_BIN jobs/sycophancy_pruning/pruning_bonham_20260918/campaign.py \
  allocate-manifests --result-root "$RESULT_ROOT" --model-key gemma4_12b \
  --gemma-balanced-amendment
```

That path still requires 512 distinct behavior-qualified questions, exact 256/256
dataset, turn-format, and bias-type marginals, and 64 examples for each
bias-type/template pair. It minimizes the maximum deviation of the eight
dataset-by-turn-by-bias cells from 64 and records the amendment identifier and
realized cell counts in the authenticated manifest receipt. The accompanying
source-bank fallback moves one ARC initially-correct quantified-reliability slot
from source template 0 to source template 1 without changing any source-family
total.

For accelerated state-sequence execution, `gpu_eval_states.sbatch` accepts
`EVALUATION_FAMILY_SET=paper_core` (generalization, useful assertions, and the
source-attribution sweep), `EVALUATION_FAMILY_SET=source_attribution`, or
`EVALUATION_FAMILY_SET=capabilities`.  The default `all` runs all four
families. Splitting the presets changes only scheduling: every frozen cell is
still required by evaluation validation and the final audit.

`accelerate_tail.sh` is a restartable submission supervisor for the constrained
`gpu_test` queue. It runs the paper-core waves first, then capabilities,
EvalPlus, weight aggregation, reporting, final audit, and the authenticated
completion email. Exact job names make restarts reuse submitted work rather
than duplicate it; a failed stage stops the supervisor for diagnosis.
`accelerate_source_attribution.sh` is the corresponding restartable packed-GPU
supervisor for adding the matched source sweep to an already-running campaign.

```bash
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
