#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "$BUNDLE_DIR/../../.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
RESULT_ROOT="${RESULT_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/random_baseline}"
LOG_ROOT="${LOG_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/random_baseline}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/n/holystore01/LABS/barak_lab/Users/ishapira/hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
SYCOBENCH_SOURCE="${SYCOBENCH_SOURCE:-/n/holystore01/LABS/barak_lab/Users/ishapira/source_snapshots/postprune_capability_audit_20260726/sycobench-600}"
WIKITEXT_REVISION="${WIKITEXT_REVISION:-b08601e04326c79dfdd32d625aee71d232d685c3}"
WIKITEXT_CACHE_ROOT="${WIKITEXT_CACHE_ROOT:-$HF_CACHE_DIR/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/$WIKITEXT_REVISION}"
WIKITEXT_SOURCE_ARROW="${WIKITEXT_SOURCE_ARROW:-$WIKITEXT_CACHE_ROOT/wikitext-test.arrow}"
WIKITEXT_DATASET_INFO="${WIKITEXT_DATASET_INFO:-$WIKITEXT_CACHE_ROOT/dataset_info.json}"
WIKITEXT_INPUT="${WIKITEXT_INPUT:-$RESULT_ROOT/inputs/wikitext_2_raw_test.jsonl}"
WIKITEXT_PIN="${WIKITEXT_PIN:-$RESULT_ROOT/registry/wikitext_pin.json}"
EMAIL_TO="${EMAIL_TO:-itaishapira@g.harvard.edu}"
RUN_DATE="20260803"

export BUNDLE_DIR REPO_DIR RESULT_ROOT LOG_ROOT HF_CACHE_DIR PYTHON_BIN
export SYCOBENCH_SOURCE EMAIL_TO RUN_DATE
export WIKITEXT_REVISION WIKITEXT_CACHE_ROOT WIKITEXT_SOURCE_ARROW
export WIKITEXT_DATASET_INFO WIKITEXT_INPUT WIKITEXT_PIN
export PYTHONPATH="$BUNDLE_DIR:$REPO_DIR:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_SILENT=true
export WANDB_DIR="${WANDB_DIR:-$RESULT_ROOT/wandb}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAMPLE_BATCH_SIZE=1
export ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}"
export USE_TF=0 USE_FLAX=0

model_value() {
  local model="$1" field="$2"
  "$PYTHON_BIN" -c 'import random_baseline as r,sys; print(r.MODEL_SPECS[sys.argv[1]][sys.argv[2]])' "$model" "$field"
}

model_snapshot() {
  local model="$1" model_id revision
  model_id="$(model_value "$model" model_id)"
  revision="$(model_value "$model" revision)"
  printf '%s/models--%s/snapshots/%s\n' "$HF_CACHE_DIR" "${model_id//\//--}" "$revision"
}

TASK_START_EPOCH=""
start_task_log() {
  local stage="$1" model="$2" label="$3"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_dir="$LOG_ROOT/by_task/$model/$stage/job_$job_id"
  mkdir -p "$task_dir" "$RESULT_ROOT" "$WANDB_DIR"
  TASK_START_EPOCH="$(date +%s)"
  export TASK_START_EPOCH
  exec > >(tee -a "$task_dir/task_$task_id.out") \
       2> >(tee -a "$task_dir/task_$task_id.err" >&2)
  printf 'experiment=random_baseline\ntask_label=%s\nstage=%s\nmodel=%s\n' "$label" "$stage" "$model"
  printf 'run_name=random_baseline\nrun_directory=%s\n' "$RESULT_ROOT"
  printf 'slurm_job_id=%s\nslurm_array_job_id=%s\nslurm_array_task_id=%s\n' \
    "${SLURM_JOB_ID:-unset}" "${SLURM_ARRAY_JOB_ID:-unset}" "${SLURM_ARRAY_TASK_ID:-unset}"
  printf 'hostname=%s\nworking_directory=%s\nstart_time=%s\n' "$(hostname)" "$(pwd)" "$(date -Is)"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
}

finish_task_log() {
  local status="$1" end_epoch
  end_epoch="$(date +%s)"
  printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
    "$(date -Is)" "$status" "$((end_epoch - TASK_START_EPOCH))"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
  fi
}

print_command() {
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
}

send_progress_email() {
  local milestone="$1" subject="$2" body="$3"
  "$PYTHON_BIN" "$BUNDLE_DIR/email_progress.py" --result-root "$RESULT_ROOT" \
    --milestone "$milestone" --subject "$subject" --body "$body" --to "$EMAIL_TO"
}

failure_email() {
  local status="$1" stage="$2" model="$3"
  set +e
  send_progress_email "failure_${stage}_${model}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}" \
    "[random_baseline] FAILURE | $stage | $model" \
    "Stage $stage failed for $model with status $status. Inspect $LOG_ROOT/by_task/$model/$stage/."
}
