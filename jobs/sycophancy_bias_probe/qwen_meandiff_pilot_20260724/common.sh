#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="qwen_meandiff_pilot_20260724"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$ROOT_DIR}}"
SUBMITTED_ENV_PYTHON="${ENV_PYTHON:-}"

cd "$REPO_DIR"
if [[ ! -f "$REPO_DIR/run_random_all_intervention.py" ]]; then
  printf '%s\n' "Missing intervention entrypoint: $REPO_DIR/run_random_all_intervention.py" >&2
  exit 1
fi

RUNTIME_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "${SUBMISSION_GIT_COMMIT:-}" && "$RUNTIME_GIT_COMMIT" != "$SUBMISSION_GIT_COMMIT" ]]; then
  printf '%s\n' \
    "Runtime checkout commit mismatch: expected=$SUBMISSION_GIT_COMMIT actual=$RUNTIME_GIT_COMMIT repo=$REPO_DIR" >&2
  exit 1
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load python/3.10.9-fasrc01
if [[ ! -f "$REPO_DIR/.env" ]]; then
  printf '%s\n' "Missing .env in $REPO_DIR" >&2
  exit 1
fi
set -a
source "$REPO_DIR/.env"
set +a

if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
  unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HOME
  unset TRITON_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  unset TMPDIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME OUT_DIR LOG_ROOT
fi

source "$REPO_DIR/jobs/sycophancy_bias_probe/storage_common.sh"
configure_sycophancy_bias_storage "$BUNDLE_NAME"

ENV_PYTHON="${SUBMITTED_ENV_PYTHON:-/n/holystore01/LABS/barak_lab/Users/ishapira/python_envs/llmsknow_py310_torch220_transformers4423/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing validated intervention Python: $ENV_PYTHON" >&2
  exit 1
fi

RUNTIME_CONTRACT_PATH="$REPO_DIR/jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py"
PILOT_RUN_ID="${PILOT_RUN_ID:-qwen_meandiff_val_pilot_20260724_v1}"
PILOT_LAYERS_CSV="${PILOT_LAYERS_CSV:-3,8,13,18,23,27}"
PILOT_MAX_QUESTIONS="${PILOT_MAX_QUESTIONS:-160}"
PILOT_ALPHAS="${PILOT_ALPHAS:--4,-2,-1,-0.5,0,0.5,1,2,4}"
PILOT_CONTROL_SEEDS="${PILOT_CONTROL_SEEDS:-0,1,2}"
PILOT_MAX_BATCH_SIZE="${PILOT_MAX_BATCH_SIZE:-8}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$SYCOPHANCY_BIAS_RESULTS_DIR/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/commonsense_qa_qwen25_7b_allfamilies_probe_random_all_20260618}"
DIRECTIONS_PATH="${DIRECTIONS_PATH:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/random_all_interventions_20260722/random_all_csqa_qwen_full_20260723_v4/commonsense_qa_qwen25_7b/directions/directions.npz}"
PILOT_BASE_ROOT="${PILOT_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/qwen_meandiff_pilot_20260724}"
CELL_ROOT="${CELL_ROOT:-$PILOT_BASE_ROOT/$PILOT_RUN_ID/commonsense_qa_qwen25_7b}"
LOG_ROOT="${PILOT_LOG_ROOT:-$LOG_ROOT/$PILOT_RUN_ID}"

sycophancy_bias_reject_home_path "CELL_ROOT" "$CELL_ROOT"
sycophancy_bias_reject_home_path "LOG_ROOT" "$LOG_ROOT"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$CELL_ROOT" \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/dose" \
  "$LOG_ROOT/slurm/aggregate" \
  "$LOG_ROOT/by_task"

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

pilot_layers() {
  printf '%s\n' "$PILOT_LAYERS_CSV" | tr ',' '\n'
}

start_structured_task_log() {
  local stage="${1:?missing stage}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_dir="$LOG_ROOT/by_task/commonsense_qa_qwen25_7b/$stage/job_$job_id"
  mkdir -p "$task_dir"
  TASK_OUT="$task_dir/task_$task_id.out"
  TASK_ERR="$task_dir/task_$task_id.err"
  TASK_STARTED_EPOCH="$(date +%s)"
  exec > >(tee -a "$TASK_OUT") 2> >(tee -a "$TASK_ERR" >&2)
  printf '[task] stage=%s task_label=commonsense_qa_qwen25_7b model=Qwen/Qwen2.5-7B-Instruct dataset=commonsense_qa\n' "$stage"
  printf '[task] run_id=%s cell_root=%s source_run_dir=%s directions=%s\n' \
    "$PILOT_RUN_ID" "$CELL_ROOT" "$SOURCE_RUN_DIR" "$DIRECTIONS_PATH"
  printf '[task] layers=%s max_questions=%s alphas=%s control_seeds=%s max_batch_size=%s\n' \
    "$PILOT_LAYERS_CSV" "$PILOT_MAX_QUESTIONS" "$PILOT_ALPHAS" \
    "$PILOT_CONTROL_SEEDS" "$PILOT_MAX_BATCH_SIZE"
  printf '[task] slurm_job_id=%s slurm_array_job_id=%s slurm_array_task_id=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}"
  printf '[task] hostname=%s working_directory=%s start_time=%s\n' \
    "$(hostname)" "$PWD" "$(iso_now)"
  printf '[task] python=%s hf_cache_dir=%s repo_dir=%s submitted_commit=%s runtime_commit=%s\n' \
    "$ENV_PYTHON" "$HF_HUB_CACHE" "$REPO_DIR" "${SUBMISSION_GIT_COMMIT:-}" "$RUNTIME_GIT_COMMIT"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  trap finish_structured_task_log EXIT
}

finish_structured_task_log() {
  local exit_code=$?
  local ended_epoch elapsed
  ended_epoch="$(date +%s)"
  elapsed=$((ended_epoch - TASK_STARTED_EPOCH))
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,Elapsed,MaxRSS,AveRSS,MaxVMSize,AveCPU || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  printf '[task] end_time=%s exit_status=%s elapsed_seconds=%s\n' \
    "$(iso_now)" "$exit_code" "$elapsed"
  trap - EXIT
  exit "$exit_code"
}

print_command() {
  printf '[task] command='
  printf '%q ' "$@"
  printf '\n'
}

export BUNDLE_NAME ROOT_DIR REPO_DIR RUNTIME_GIT_COMMIT SUBMISSION_GIT_COMMIT
export ENV_PYTHON RUNTIME_CONTRACT_PATH SOURCE_RUN_DIR DIRECTIONS_PATH
export PILOT_RUN_ID PILOT_LAYERS_CSV PILOT_MAX_QUESTIONS PILOT_ALPHAS
export PILOT_CONTROL_SEEDS PILOT_MAX_BATCH_SIZE PILOT_BASE_ROOT CELL_ROOT LOG_ROOT

