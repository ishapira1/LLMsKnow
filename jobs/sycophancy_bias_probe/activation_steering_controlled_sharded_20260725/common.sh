#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="activation_steering_controlled_sharded_20260725"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/n/home12/ishapira/LLMsKnow}}"
SUBMITTED_ENV_PYTHON="${ENV_PYTHON:-}"
cd "$REPO_DIR"

if [[ ! -f run_activation_steering.py ]]; then
  printf '%s\n' "Missing controlled activation-steering entrypoint in $REPO_DIR" >&2
  exit 1
fi

RUNTIME_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "${SUBMISSION_GIT_COMMIT:-}" && "$RUNTIME_GIT_COMMIT" != "$SUBMISSION_GIT_COMMIT" ]]; then
  printf '%s\n' \
    "Runtime commit mismatch expected=$SUBMISSION_GIT_COMMIT actual=$RUNTIME_GIT_COMMIT" >&2
  exit 1
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load python/3.10.9-fasrc01
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
LOG_ROOT="${ACTIVATION_STEERING_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/$BUNDLE_NAME}"

RUNTIME_ENV_DIR="${ACTIVATION_STEERING_ENV_DIR:-$SYCOPHANCY_STORAGE_ROOT/python_envs/llmsknow_py310_torch220_transformers4432}"
ENV_PYTHON="${SUBMITTED_ENV_PYTHON:-$RUNTIME_ENV_DIR/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing controlled intervention Python: $ENV_PYTHON" >&2
  exit 1
fi
RUNTIME_CONTRACT_PATH="$REPO_DIR/jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/runtime_contract.py"
validate_runtime_contract() {
  "$ENV_PYTHON" "$RUNTIME_CONTRACT_PATH" "$@"
}
validate_runtime_contract

CONFIG_PATH="${ACTIVATION_STEERING_CONFIG:-$REPO_DIR/configs/experiments/activation_steering_controlled_20260725.json}"
QUESTION_MANIFEST="${QUESTION_MANIFEST:-$REPO_DIR/configs/experiments/activation_steering_audited_1000_20260725.jsonl}"
PREFLIGHT_MANIFEST="${PREFLIGHT_MANIFEST:-$REPO_DIR/configs/experiments/activation_steering_preflight_8_20260725.jsonl}"
ALPACA_UTILITY_MANIFEST="${ALPACA_UTILITY_MANIFEST:-$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/evaluation/alpaca_utility.jsonl}"
SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-$SYCOPHANCY_BIAS_RESULTS_DIR}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-manual_controlled_20260725}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_controlled_20260725}"
CONFIG_HASH="$("$ENV_PYTHON" -c 'import hashlib,pathlib,sys; c=pathlib.Path(sys.argv[1]); m=pathlib.Path(sys.argv[2]); payload=c.read_bytes()+b"\0"+(m.read_bytes() if m.is_file() else ("<missing:"+str(m)+">").encode()); print(hashlib.sha256(payload).hexdigest()[:12])' "$CONFIG_PATH" "$QUESTION_MANIFEST")"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/${EXPERIMENT_RUN_ID}_$CONFIG_HASH}"
HF_CACHE_DIR="$HF_HUB_CACHE"
sycophancy_bias_reject_home_path "INTERVENTION_ROOT" "$INTERVENTION_ROOT"

mkdir -p \
  "$INTERVENTION_ROOT" \
  "$LOG_ROOT/by_task" \
  "$LOG_ROOT/slurm/inspection" \
  "$LOG_ROOT/slurm/tiny" \
  "$LOG_ROOT/slurm/validation" \
  "$LOG_ROOT/slurm/fit" \
  "$LOG_ROOT/slurm/screen" \
  "$LOG_ROOT/slurm/selection" \
  "$LOG_ROOT/slurm/dose" \
  "$LOG_ROOT/slurm/test" \
  "$LOG_ROOT/slurm/probe" \
  "$LOG_ROOT/slurm/transfer" \
  "$LOG_ROOT/slurm/geometry" \
  "$LOG_ROOT/slurm/alpaca" \
  "$LOG_ROOT/slurm/aggregate"

MODEL_KEYS=(llama31_8b qwen25_7b)
MODEL_SLUGS=(meta_llama_Llama_3_1_8B_Instruct Qwen_Qwen2_5_7B_Instruct)
MODEL_IDENTIFIERS=(meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct)
MODEL_NONTERMINAL_COUNTS=(31 27)
DATASET_NAMES=(commonsense_qa arc_challenge)
SOURCE_RUN_NAMES_LLAMA=(
  commonsense_qa_llama31_8b_allfamilies_probe_random_all_20260618
  arc_challenge_llama31_8b_allfamilies_probe_random_all_20260618
)
SOURCE_RUN_NAMES_QWEN=(
  commonsense_qa_qwen25_7b_allfamilies_probe_random_all_20260618
  arc_challenge_qwen25_7b_allfamilies_probe_random_all_20260618
)

source_run_dir() {
  local model_index="${1:?model index}" dataset_index="${2:?dataset index}"
  local run_name
  if [[ "$model_index" == "0" ]]; then
    run_name="${SOURCE_RUN_NAMES_LLAMA[$dataset_index]}"
  else
    run_name="${SOURCE_RUN_NAMES_QWEN[$dataset_index]}"
  fi
  printf '%s\n' \
    "$SOURCE_RESULTS_ROOT/${MODEL_SLUGS[$model_index]}/${DATASET_NAMES[$dataset_index]}/$run_name"
}

model_selected() {
  local model_index="${1:?model index}" filter="${TASK_FILTER:-}"
  [[ -z "$filter" ]] && return 0
  [[ ",$filter," == *",${MODEL_KEYS[$model_index]},"* ]]
}

resolve_model() {
  MODEL_INDEX="${1:?model index}"
  MODEL_KEY="${MODEL_KEYS[$MODEL_INDEX]}"
  MODEL_SLUG="${MODEL_SLUGS[$MODEL_INDEX]}"
  MODEL_IDENTIFIER="${MODEL_IDENTIFIERS[$MODEL_INDEX]}"
  NONTERMINAL_LAYER_COUNT="${MODEL_NONTERMINAL_COUNTS[$MODEL_INDEX]}"
  DIRECTIONS_DIR="$INTERVENTION_ROOT/directions/$MODEL_KEY"
  DIRECTIONS_PATH="$DIRECTIONS_DIR/directions.npz"
  MODEL_ROOT="$INTERVENTION_ROOT/models/$MODEL_KEY"
}

resolve_cell() {
  CELL_INDEX="${1:?cell index}"
  MODEL_INDEX=$((CELL_INDEX % 2))
  DATASET_INDEX=$((CELL_INDEX / 2))
  resolve_model "$MODEL_INDEX"
  DATASET_NAME="${DATASET_NAMES[$DATASET_INDEX]}"
  SOURCE_RUN_DIR="$(source_run_dir "$MODEL_INDEX" "$DATASET_INDEX")"
  TASK_LABEL="${DATASET_NAME}_${MODEL_KEY}"
  CELL_ROOT="$MODEL_ROOT/$DATASET_NAME"
}

resolve_layer_task() {
  local task_id="${1:?task id}" offset=0 model_index dataset_index count
  for dataset_index in 0 1; do
    for model_index in 0 1; do
      count="${MODEL_NONTERMINAL_COUNTS[$model_index]}"
      if (( task_id < offset + count )); then
        resolve_cell $((dataset_index * 2 + model_index))
        LAYER=$((task_id - offset + 1))
        return 0
      fi
      offset=$((offset + count))
    done
  done
  return 1
}

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

print_command() {
  printf '[task] command='
  printf '%q ' "$@"
  printf '\n'
}

start_structured_task_log() {
  local stage="${1:?stage}" probe_family="${2:-probe_bias_random_all}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_name="${TASK_LABEL:-${MODEL_KEY:-controlled}}"
  local task_dir="$LOG_ROOT/by_task/$task_name/$stage/job_$job_id"
  mkdir -p "$task_dir"
  TASK_STARTED_EPOCH="$(date +%s)"
  exec > >(tee -a "$task_dir/task_$task_id.out") \
    2> >(tee -a "$task_dir/task_$task_id.err" >&2)
  printf '[task] stage=%s task_label=%s model=%s dataset=%s probe_family=%s\n' \
    "$stage" "$task_name" "${MODEL_KEY:-}" "${DATASET_NAME:-}" "$probe_family"
  printf '[task] run_name=%s run_directory=%s\n' "$EXPERIMENT_RUN_ID" "$INTERVENTION_ROOT"
  printf '[task] slurm_job_id=%s array_job_id=%s array_task_id=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}"
  printf '[task] hostname=%s cwd=%s start_time=%s\n' "$(hostname)" "$PWD" "$(iso_now)"
  printf '[task] python=%s config=%s question_manifest=%s\n' \
    "$ENV_PYTHON" "$CONFIG_PATH" "$QUESTION_MANIFEST"
  printf '[task] runtime_commit=%s submission_commit=%s\n' \
    "$RUNTIME_GIT_COMMIT" "${SUBMISSION_GIT_COMMIT:-}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  trap finish_structured_task_log EXIT
}

finish_structured_task_log() {
  local exit_code=$? ended elapsed
  ended="$(date +%s)"
  elapsed=$((ended - TASK_STARTED_EPOCH))
  if command -v sstat >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sstat -j "${SLURM_JOB_ID}.batch" --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU || true
  fi
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  printf '[task] end_time=%s exit_status=%s elapsed_seconds=%s\n' \
    "$(iso_now)" "$exit_code" "$elapsed"
  trap - EXIT
  exit "$exit_code"
}

export REPO_DIR ENV_PYTHON RUNTIME_ENV_DIR RUNTIME_CONTRACT_PATH
export CONFIG_PATH QUESTION_MANIFEST PREFLIGHT_MANIFEST ALPACA_UTILITY_MANIFEST SOURCE_RESULTS_ROOT
export EXPERIMENT_RUN_ID INTERVENTION_ROOT HF_CACHE_DIR SUBMISSION_GIT_COMMIT
