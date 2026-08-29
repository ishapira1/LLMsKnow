#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd "$BUNDLE_DIR/../../.." && pwd -P)}"
RESULT_ROOT="${RESULT_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/cross_model_confidence_resistance_20260828}"
LOG_ROOT="${LOG_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/n/holystore01/LABS/barak_lab/Users/ishapira/hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
RUNNER="$REPO_ROOT/scripts/run_confidence_resistance.py"

export BUNDLE_DIR REPO_ROOT RESULT_ROOT LOG_ROOT HF_CACHE_DIR PYTHON_BIN RUNNER
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR" HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline WANDB_SILENT=true
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAMPLE_BATCH_SIZE=1
export ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}" USE_TF=0 USE_FLAX=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RESULT_ROOT/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RESULT_ROOT/.cache}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$RESULT_ROOT/.tiktoken_cache}"

TASK_START_EPOCH=""

print_command() {
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
}

start_task_log() {
  stage="$1"
  dataset_model="$2"
  task_label="$3"
  model="${4:-none}"
  dataset="${5:-none}"
  job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  task_dir="$LOG_ROOT/by_task/$dataset_model/$stage/job_$job_id"
  mkdir -p "$task_dir" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TIKTOKEN_CACHE_DIR"
  TASK_START_EPOCH="$(date +%s)"
  exec > >(tee -a "$task_dir/task_$task_id.out") 2> >(tee -a "$task_dir/task_$task_id.err" >&2)
  printf 'experiment=%s\nstage=%s\ntask_label=%s\nmodel=%s\ndataset=%s\n' \
    cross_model_confidence_resistance_20260828 "$stage" "$task_label" "$model" "$dataset"
  printf 'run_name=%s\nrun_directory=%s\nresult_root=%s\nlog_root=%s\n' \
    cross_model_confidence_resistance_20260828 "$RESULT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
  printf 'slurm_job_id=%s\nslurm_array_job_id=%s\nslurm_array_task_id=%s\n' \
    "${SLURM_JOB_ID:-unset}" "${SLURM_ARRAY_JOB_ID:-unset}" "${SLURM_ARRAY_TASK_ID:-unset}"
  printf 'hostname=%s\nworking_directory=%s\nstart_time=%s\n' \
    "$(hostname)" "$(pwd)" "$(date -Is)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
}

finish_task_log() {
  status="$1"
  end_epoch="$(date +%s)"
  printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
    "$(date -Is)" "$status" "$((end_epoch-TASK_START_EPOCH))"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
  fi
}

require_runner() {
  [[ -f "$RUNNER" ]] || { printf 'missing runner: %s\n' "$RUNNER" >&2; return 2; }
  [[ "$ALLOW_STALE_LOCK_CLEANUP" == 0 ]] || {
    printf 'This experiment never deletes .run.lock files; ALLOW_STALE_LOCK_CLEANUP must remain 0.\n' >&2
    return 2
  }
}

require_h200() {
  "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; p=torch.cuda.get_device_properties(0); print(f"gpu_name={p.name} gpu_gib={p.total_memory/1024**3:.2f}"); assert p.total_memory/1024**3 >= 70, "H200-class memory required"'
}
