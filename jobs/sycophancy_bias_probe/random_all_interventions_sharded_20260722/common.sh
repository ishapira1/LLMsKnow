#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="random_all_interventions_sharded_20260722"
JOB_DATE_TAG="20260722"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load python/3.10.9-fasrc01

ENV_PYTHON="${ENV_PYTHON:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing python interpreter: $ENV_PYTHON" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $REPO_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
  unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HOME
  unset TRITON_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  unset TMPDIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME OUT_DIR LOG_ROOT
fi

source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"

SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-$SYCOPHANCY_BIAS_RESULTS_DIR}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-manual_20260722}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/random_all_interventions_20260722}"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/$EXPERIMENT_RUN_ID}"
sycophancy_bias_reject_home_path "INTERVENTION_ROOT" "$INTERVENTION_ROOT"
HF_CACHE_DIR="$HF_HUB_CACHE"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  "$TRITON_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$TMPDIR" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$INTERVENTION_ROOT" \
  "$LOG_ROOT" \
  "$LOG_ROOT/by_task"

TASK_LABELS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)
MODEL_SLUGS=(
  meta_llama_Llama_3_1_8B_Instruct
  Qwen_Qwen2_5_7B_Instruct
  meta_llama_Llama_3_1_8B_Instruct
  Qwen_Qwen2_5_7B_Instruct
)
DATASET_NAMES=(
  commonsense_qa
  commonsense_qa
  arc_challenge
  arc_challenge
)
SOURCE_RUN_NAMES=(
  commonsense_qa_llama31_8b_allfamilies_probe_random_all_20260618
  commonsense_qa_qwen25_7b_allfamilies_probe_random_all_20260618
  arc_challenge_llama31_8b_allfamilies_probe_random_all_20260618
  arc_challenge_qwen25_7b_allfamilies_probe_random_all_20260618
)
NONTERMINAL_LAYER_COUNTS=(31 27 31 27)

selected_base_indices() {
  local index
  for index in "${!TASK_LABELS[@]}"; do
    if [[ -z "${TASK_FILTER:-}" || "${TASK_FILTER:-}" == "${TASK_LABELS[$index]}" ]]; then
      printf '%s\n' "$index"
    fi
  done
}

resolve_task_context() {
  local base_index="${1:?missing base index}"
  TASK_LABEL="${TASK_LABELS[$base_index]}"
  MODEL_SLUG="${MODEL_SLUGS[$base_index]}"
  DATASET_NAME="${DATASET_NAMES[$base_index]}"
  SOURCE_RUN_NAME="${SOURCE_RUN_NAMES[$base_index]}"
  NONTERMINAL_LAYER_COUNT="${NONTERMINAL_LAYER_COUNTS[$base_index]}"
  SOURCE_RUN_DIR="$SOURCE_RESULTS_ROOT/$MODEL_SLUG/$DATASET_NAME/$SOURCE_RUN_NAME"
  CELL_ROOT="$INTERVENTION_ROOT/$TASK_LABEL"
  DIRECTIONS_DIR="$CELL_ROOT/directions"
  DIRECTIONS_PATH="$DIRECTIONS_DIR/directions.npz"
  SELECTION_JSON="$CELL_ROOT/selected_intervention.json"
  mkdir -p "$CELL_ROOT"
  if [[ ! -d "$SOURCE_RUN_DIR" ]]; then
    printf '%s\n' "Missing source random_all run: $SOURCE_RUN_DIR" >&2
    exit 1
  fi
}

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

start_structured_task_log() {
  local stage="${1:?missing stage}"
  local probe_family="${2:-none}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_dir="$LOG_ROOT/by_task/$TASK_LABEL/$stage/job_$job_id"
  mkdir -p "$task_dir"
  TASK_OUT="$task_dir/task_$task_id.out"
  TASK_ERR="$task_dir/task_$task_id.err"
  TASK_STARTED_EPOCH="$(date +%s)"
  exec > >(tee -a "$TASK_OUT") 2> >(tee -a "$TASK_ERR" >&2)
  printf '[task] stage=%s task_label=%s model_slug=%s dataset=%s probe_family=%s\n' \
    "$stage" "$TASK_LABEL" "$MODEL_SLUG" "$DATASET_NAME" "$probe_family"
  printf '[task] source_run_name=%s source_run_dir=%s cell_root=%s\n' \
    "$SOURCE_RUN_NAME" "$SOURCE_RUN_DIR" "$CELL_ROOT"
  printf '[task] slurm_job_id=%s slurm_array_job_id=%s slurm_array_task_id=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}"
  printf '[task] hostname=%s working_directory=%s start_time=%s\n' \
    "$(hostname)" "$PWD" "$(iso_now)"
  printf '[task] python=%s hf_cache_dir=%s intervention_root=%s\n' \
    "$ENV_PYTHON" "$HF_CACHE_DIR" "$INTERVENTION_ROOT"
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

export SOURCE_RESULTS_ROOT EXPERIMENT_RUN_ID INTERVENTION_BASE_ROOT INTERVENTION_ROOT HF_CACHE_DIR
