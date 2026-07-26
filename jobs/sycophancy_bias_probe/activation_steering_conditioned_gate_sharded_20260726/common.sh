#!/usr/bin/env bash

set -euo pipefail

BUNDLE_NAME="activation_steering_conditioned_gate_sharded_20260726"
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

CONFIG_PATH="${ACTIVATION_STEERING_CONFIG:-$REPO_DIR/configs/experiments/activation_steering_conditioned_gate_20260726.json}"
QUESTION_MANIFEST="${QUESTION_MANIFEST:-$REPO_DIR/configs/experiments/activation_steering_signal_300_20260726.jsonl}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-activation_steering_conditioned_gate_20260726_v1}"
SOURCE_INTERVENTION_RUN="${SOURCE_INTERVENTION_RUN:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_signal_20260726/activation_steering_signal_20260726_v1_05d7b1eae414}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/activation_steering_conditioned_gate_20260726}"
CONFIG_HASH="$("$ENV_PYTHON" -c 'import hashlib,pathlib,sys; c=pathlib.Path(sys.argv[1]); m=pathlib.Path(sys.argv[2]); print(hashlib.sha256(c.read_bytes()+b"\\0"+m.read_bytes()).hexdigest()[:12])' "$CONFIG_PATH" "$QUESTION_MANIFEST")"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/${EXPERIMENT_RUN_ID}_$CONFIG_HASH}"
AUDIT_OUTPUT_DIR="$INTERVENTION_ROOT/stage_a_audit"
LOG_ROOT="${ACTIVATION_STEERING_LOG_ROOT:-$REPO_DIR/jobs/sycophancy_bias_probe/logs/$BUNDLE_NAME}"

SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-$SYCOPHANCY_BIAS_RESULTS_DIR}"
LLAMA_CSQA_SOURCE="$SOURCE_RESULTS_ROOT/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/commonsense_qa_llama31_8b_allfamilies_probe_random_all_20260618"
LLAMA_ARC_SOURCE="$SOURCE_RESULTS_ROOT/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/arc_challenge_llama31_8b_allfamilies_sampling_20260618"
QWEN_CSQA_SOURCE="$SOURCE_RESULTS_ROOT/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/commonsense_qa_qwen25_7b_allfamilies_probe_random_all_20260618"
QWEN_ARC_SOURCE="$SOURCE_RESULTS_ROOT/Qwen_Qwen2_5_7B_Instruct/arc_challenge/arc_challenge_qwen25_7b_allfamilies_sampling_20260618"
LLAMA_DIRECTIONS="$SOURCE_INTERVENTION_RUN/directions/llama31_8b/directions.npz"
QWEN_DIRECTIONS="$SOURCE_INTERVENTION_RUN/directions/qwen25_7b/directions.npz"

mkdir -p \
  "$INTERVENTION_ROOT" \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/audit" \
  "$LOG_ROOT/slurm/cohort" \
  "$LOG_ROOT/slurm/bf16" \
  "$LOG_ROOT/slurm/projection" \
  "$LOG_ROOT/slurm/validation" \
  "$LOG_ROOT/slurm/selection" \
  "$LOG_ROOT/slurm/test" \
  "$LOG_ROOT/slurm/control" \
  "$LOG_ROOT/slurm/sensitivity" \
  "$LOG_ROOT/slurm/aggregate" \
  "$LOG_ROOT/by_task/stage_a/audit"

MODEL_KEYS=(llama31_8b qwen25_7b)
MODEL_IDENTIFIERS=(
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
)
ARC_SOURCES=("$LLAMA_ARC_SOURCE" "$QWEN_ARC_SOURCE")

resolve_conditioned_model() {
  MODEL_INDEX="${1:?model index}"
  MODEL_KEY="${MODEL_KEYS[$MODEL_INDEX]}"
  MODEL_IDENTIFIER="${MODEL_IDENTIFIERS[$MODEL_INDEX]}"
  ARC_SOURCE="${ARC_SOURCES[$MODEL_INDEX]}"
  COHORT_MANIFEST="$INTERVENTION_ROOT/cohorts/${MODEL_KEY}_arc_neutral_correct.jsonl"
  CONDITIONED_DIRECTIONS="$AUDIT_OUTPUT_DIR/conditioned_directions_model_${MODEL_INDEX}/directions.npz"
  MODEL_ROOT="$INTERVENTION_ROOT/stage_b/$MODEL_KEY"
  TASK_LABEL="${MODEL_KEY}_arc"
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
  local task_dir="$LOG_ROOT/by_task/${TASK_LABEL:-stage_a}/$stage/job_$job_id"
  mkdir -p "$task_dir"
  TASK_STARTED_EPOCH="$(date +%s)"
  exec > >(tee -a "$task_dir/task_0.out") \
    2> >(tee -a "$task_dir/task_0.err" >&2)
  printf '[task] stage=%s task_label=%s model=%s dataset=%s\n' \
    "$stage" "${TASK_LABEL:-mean_cancellation_audit}" \
    "${MODEL_IDENTIFIER:-both}" "${DATASET_NAME:-arc_challenge,commonsense_qa}"
  printf '[task] run_name=%s run_directory=%s\n' "$EXPERIMENT_RUN_ID" "$INTERVENTION_ROOT"
  printf '[task] slurm_job_id=%s hostname=%s cwd=%s start_time=%s\n' \
    "${SLURM_JOB_ID:-}" "$(hostname)" "$PWD" "$(iso_now)"
  printf '[task] python=%s config=%s question_manifest=%s\n' \
    "$ENV_PYTHON" "$CONFIG_PATH" "$QUESTION_MANIFEST"
  printf '[task] runtime_commit=%s submission_commit=%s\n' \
    "$RUNTIME_GIT_COMMIT" "${SUBMISSION_GIT_COMMIT:-}"
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

export REPO_DIR ENV_PYTHON CONFIG_PATH QUESTION_MANIFEST EXPERIMENT_RUN_ID
export INTERVENTION_ROOT AUDIT_OUTPUT_DIR SOURCE_INTERVENTION_RUN LOG_ROOT
export LLAMA_CSQA_SOURCE LLAMA_ARC_SOURCE QWEN_CSQA_SOURCE QWEN_ARC_SOURCE
export LLAMA_DIRECTIONS QWEN_DIRECTIONS RUNTIME_GIT_COMMIT
