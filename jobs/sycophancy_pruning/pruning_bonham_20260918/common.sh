#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd "$BUNDLE_DIR/../../.." && pwd -P)}"
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

CPU_PYTHON_BIN="${CPU_PYTHON_BIN:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
GEMMA_PYTHON_BIN="${GEMMA_PYTHON_BIN:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_runtimes/gemma4_12b_20260911/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$CPU_PYTHON_BIN}"
RESULT_ROOT="${RESULT_ROOT:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/pruning_bonham_20260918}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/n/holystore01/LABS/barak_lab/Users/ishapira/hf_cache}"
SUITE_SOURCE_BINDINGS="${SUITE_SOURCE_BINDINGS:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/causal_eval_refactor_20260810/suite_sources_v4_59a4b5199df24309_20260812T183000Z_9363/suite_source_bindings.json}"
CONFIG_PATH="$REPO_DIR/configs/experiments/pruning_bonham_20260918.json"
CAPABILITY_SOURCE_ROOT="$RESULT_ROOT/sources/capabilities"
EVALPLUS_IMAGE="${EVALPLUS_IMAGE:-/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_results/causal_eval_refactor_20260810/frozen_containers/evalplus_26d6d00bb1fd0fa37f39c99d5290da67891d1c5e_py311_amd64/evalplus.sif}"
EVALPLUS_IMAGE_SHA256="468d3609154942b83f5c1ea8b33032b4de5be8b13b05285f78bfeea799433a5e"
ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}"

export BUNDLE_DIR REPO_DIR CPU_PYTHON_BIN GEMMA_PYTHON_BIN PYTHON_BIN
export RESULT_ROOT LOG_ROOT HF_CACHE_DIR SUITE_SOURCE_BINDINGS CONFIG_PATH
export CAPABILITY_SOURCE_ROOT EVALPLUS_IMAGE EVALPLUS_IMAGE_SHA256 ALLOW_STALE_LOCK_CLEANUP
export PYTHONPATH="$BUNDLE_DIR:$REPO_DIR:$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub" TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline WANDB_SILENT=true PYTHONDONTWRITEBYTECODE=1
export USE_TF=0 USE_FLAX=0
export SAMPLE_BATCH_SIZE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_for_model() {
  if [[ "${1:-}" == gemma4_12b ]]; then
    printf '%s\n' "$GEMMA_PYTHON_BIN"
  else
    printf '%s\n' "$CPU_PYTHON_BIN"
  fi
}

require_runtime() {
  [[ -x "$CPU_PYTHON_BIN" ]] || { printf 'missing Harvard Python: %s\n' "$CPU_PYTHON_BIN" >&2; return 2; }
  [[ -x "$GEMMA_PYTHON_BIN" ]] || { printf 'missing Gemma Python: %s\n' "$GEMMA_PYTHON_BIN" >&2; return 2; }
  [[ -f "$SUITE_SOURCE_BINDINGS" ]] || { printf 'missing source binding: %s\n' "$SUITE_SOURCE_BINDINGS" >&2; return 2; }
  [[ -f "$CONFIG_PATH" ]] || { printf 'missing Bonham config: %s\n' "$CONFIG_PATH" >&2; return 2; }
  [[ "$ALLOW_STALE_LOCK_CLEANUP" == 0 ]] || {
    printf 'Bonham never removes .run.lock files automatically; leave ALLOW_STALE_LOCK_CLEANUP=0.\n' >&2
    return 2
  }
}

TASK_START_EPOCH=""
start_task_log() {
  stage="$1"; model_key="${2:-shared}"; task_label="${3:-none}"
  job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  task_id="${SLURM_ARRAY_TASK_ID:-0}"
  task_dir="$LOG_ROOT/by_task/${model_key}/${stage}/job_${job_id}"
  mkdir -p "$task_dir" "$LOG_ROOT/submit" "$LOG_ROOT/slurm/$stage"
  TASK_START_EPOCH="$(date +%s)"; export TASK_START_EPOCH
  exec > >(tee -a "$task_dir/task_${task_id}.out") 2> >(tee -a "$task_dir/task_${task_id}.err" >&2)
  printf 'experiment=pruning_bonham_20260918\nstage=%s\ntask_label=%s\nmodel=%s\n' "$stage" "$task_label" "$model_key"
  printf 'result_root=%s\nslurm_job_id=%s\nslurm_array_job_id=%s\nslurm_array_task_id=%s\n' \
    "$RESULT_ROOT" "${SLURM_JOB_ID:-unset}" "${SLURM_ARRAY_JOB_ID:-unset}" "${SLURM_ARRAY_TASK_ID:-unset}"
  printf 'hostname=%s\nworking_directory=%s\npython=%s\nstart_time=%s\n' \
    "$(hostname)" "$(pwd)" "$PYTHON_BIN" "$(date -Is)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
}

finish_task_log() {
  status="$1"; end_epoch="$(date +%s)"
  printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
    "$(date -Is)" "$status" "$((end_epoch-TASK_START_EPOCH))"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
  fi
}

print_command() { printf 'command='; printf '%q ' "$@"; printf '\n'; }
