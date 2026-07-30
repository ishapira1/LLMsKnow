#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="activation_steering_multilayer_gate_sharded_20260729"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/n/home12/ishapira/LLMsKnow_activation_signal_20260726}}"
SUBMITTED_ENV_PYTHON="${ENV_PYTHON:-}"
cd "$REPO_DIR"

test -f run_activation_steering.py || {
  printf 'Missing activation-steering entrypoint in %s\n' "$REPO_DIR" >&2
  exit 1
}
RUNTIME_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ -n "${SUBMISSION_GIT_COMMIT:-}" && "$RUNTIME_GIT_COMMIT" != "$SUBMISSION_GIT_COMMIT" ]]; then
  printf 'Runtime commit mismatch expected=%s actual=%s\n' \
    "$SUBMISSION_GIT_COMMIT" "$RUNTIME_GIT_COMMIT" >&2
  exit 1
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load python/3.10.9-fasrc01
test -f .env || {
  printf 'Missing .env in %s\n' "$REPO_DIR" >&2
  exit 1
}
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

RUNTIME_ENV_DIR="${ACTIVATION_STEERING_ENV_DIR:-$SYCOPHANCY_STORAGE_ROOT/python_envs/llmsknow_py310_torch220_transformers4432}"
ENV_PYTHON="${SUBMITTED_ENV_PYTHON:-$RUNTIME_ENV_DIR/bin/python}"
test -x "$ENV_PYTHON" || {
  printf 'Missing runtime Python: %s\n' "$ENV_PYTHON" >&2
  exit 1
}

CONFIG_PATH="${ACTIVATION_STEERING_CONFIG:-$REPO_DIR/configs/experiments/activation_steering_multilayer_gate_20260729.json}"
SOURCE_CONDITIONED_ROOT="${SOURCE_CONDITIONED_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_conditioned_gate_20260726/activation_steering_conditioned_gate_20260726_v5_38918a643f45}"
SOURCE_DECISION="$SOURCE_CONDITIONED_ROOT/stage_a_audit/decision.json"
test -s "$SOURCE_DECISION"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-activation_steering_multilayer_gate_20260729_v1}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_multilayer_gate_20260729}"
CONFIG_HASH="$("$ENV_PYTHON" -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()+b"\0"+pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()[:12])' "$CONFIG_PATH" "$SOURCE_DECISION")"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/${EXPERIMENT_RUN_ID}_$CONFIG_HASH}"
LOG_ROOT="${ACTIVATION_STEERING_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/$BUNDLE_NAME}"

SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-$SYCOPHANCY_BIAS_RESULTS_DIR}"
LLAMA_ARC_SOURCE="$SOURCE_RESULTS_ROOT/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/arc_challenge_llama31_8b_allfamilies_sampling_20260618"
QWEN_ARC_SOURCE="$SOURCE_RESULTS_ROOT/Qwen_Qwen2_5_7B_Instruct/arc_challenge/arc_challenge_qwen25_7b_allfamilies_sampling_20260618"

mkdir -p \
  "$INTERVENTION_ROOT/cohorts" \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/bf16" \
  "$LOG_ROOT/slurm/validation" \
  "$LOG_ROOT/slurm/selection"

MODEL_KEYS=(llama31_8b qwen25_7b)
MODEL_IDENTIFIERS=(
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
)
ARC_SOURCES=("$LLAMA_ARC_SOURCE" "$QWEN_ARC_SOURCE")

resolve_multilayer_model() {
  MODEL_INDEX="${1:?model index}"
  MODEL_KEY="${MODEL_KEYS[$MODEL_INDEX]}"
  MODEL_IDENTIFIER="${MODEL_IDENTIFIERS[$MODEL_INDEX]}"
  ARC_SOURCE="${ARC_SOURCES[$MODEL_INDEX]}"
  COHORT_MANIFEST="$SOURCE_CONDITIONED_ROOT/cohorts/${MODEL_KEY}_arc_neutral_correct.jsonl"
  CONDITIONED_DIRECTIONS="$SOURCE_CONDITIONED_ROOT/stage_a_audit/conditioned_directions_model_${MODEL_INDEX}/directions.npz"
  MODEL_ROOT="$INTERVENTION_ROOT/models/$MODEL_KEY"
  TASK_LABEL="${MODEL_KEY}_arc_multilayer"
  read -r PRIMARY_FAMILY SELECTED_LAYER < <(
    "$ENV_PYTHON" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); m=d["models"][sys.argv[2]]; assert d["gpu_stage_authorized"] and m["primary_family"]; print(m["primary_family"],m["nominated_layer"])' \
      "$SOURCE_DECISION" "$MODEL_IDENTIFIER"
  )
  test -s "$COHORT_MANIFEST"
  test -s "$CONDITIONED_DIRECTIONS"
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
  local stage="${1:?stage}"
  local job_id="${SLURM_JOB_ID:-local}"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  local task_dir="$LOG_ROOT/by_task/${TASK_LABEL:-aggregate}/$stage/job_$job_id"
  mkdir -p "$task_dir"
  TASK_STARTED_EPOCH="$(date +%s)"
  exec > >(tee -a "$task_dir/task_${task_id}.out") \
    2> >(tee -a "$task_dir/task_${task_id}.err" >&2)
  printf '[task] stage=%s task_label=%s model=%s dataset=arc_challenge\n' \
    "$stage" "${TASK_LABEL:-aggregate}" "${MODEL_IDENTIFIER:-both}"
  printf '[task] run_name=%s run_directory=%s\n' "$EXPERIMENT_RUN_ID" "$INTERVENTION_ROOT"
  printf '[task] slurm_job_id=%s array_job_id=%s array_task_id=%s hostname=%s cwd=%s start_time=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}" \
    "$(hostname)" "$PWD" "$(iso_now)"
  printf '[task] python=%s config=%s runtime_commit=%s submission_commit=%s\n' \
    "$ENV_PYTHON" "$CONFIG_PATH" "$RUNTIME_GIT_COMMIT" "${SUBMISSION_GIT_COMMIT:-}"
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
  printf '[task] end_time=%s exit_status=%s elapsed_seconds=%s\n' \
    "$(iso_now)" "$exit_code" "$elapsed"
  trap - EXIT
  exit "$exit_code"
}

export REPO_DIR ENV_PYTHON CONFIG_PATH SOURCE_CONDITIONED_ROOT SOURCE_DECISION
export EXPERIMENT_RUN_ID INTERVENTION_ROOT LOG_ROOT RUNTIME_GIT_COMMIT
export LLAMA_ARC_SOURCE QWEN_ARC_SOURCE
