#!/bin/bash
set -euo pipefail

BUNDLE_NAME="anti_sycophancy_request_sharded_20260623"
JOB_DATE_TAG="20260623"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $ROOT_DIR" >&2
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

SLURM_LOG_ROOT="$LOG_ROOT/slurm"
SAMPLING_SLURM_LOG_DIR="$SLURM_LOG_ROOT/sampling"
ANALYSIS_SLURM_LOG_DIR="$SLURM_LOG_ROOT/analysis"
SUMMARY_SLURM_LOG_DIR="$SLURM_LOG_ROOT/summary"
STRUCTURED_LOG_ROOT="$LOG_ROOT/by_task"
SUBMIT_LOG_DIR="$LOG_ROOT/submit"
mkdir -p "$SAMPLING_SLURM_LOG_DIR" "$ANALYSIS_SLURM_LOG_DIR" "$SUMMARY_SLURM_LOG_DIR" "$STRUCTURED_LOG_ROOT" "$SUBMIT_LOG_DIR"

SUBMISSION_STEM="$(date +%Y%m%dT%H%M%S%z)_pid_$$"
SUBMIT_LOG_FILE="$SUBMIT_LOG_DIR/submit_${SUBMISSION_STEM}.log"
SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/submission_${SUBMISSION_STEM}.env"
LATEST_SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/latest_submission.env"
SAMPLING_TASK_MATRIX="$SUBMIT_LOG_DIR/sampling_task_matrix_${SUBMISSION_STEM}.tsv"
ANALYSIS_TASK_MATRIX="$SUBMIT_LOG_DIR/analysis_task_matrix_${SUBMISSION_STEM}.tsv"

log_line() {
  printf '%s\n' "$1"
  printf '%s\n' "$1" >> "$SUBMIT_LOG_FILE"
}

log_cmd() {
  local prefix="$1"
  shift
  printf '%s' "$prefix"
  printf '%s' "$prefix" >> "$SUBMIT_LOG_FILE"
  printf '%q ' "$@"
  printf '%q ' "$@" >> "$SUBMIT_LOG_FILE"
  printf '\n'
  printf '\n' >> "$SUBMIT_LOG_FILE"
}

slugify() {
  "${ENV_PYTHON_FOR_REGISTRY:-python}" - "$1" <<'PY'
import sys
value = sys.argv[1]
cleaned = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
print(cleaned or "model")
PY
}

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log_line "[submit-${JOB_DATE_TAG}] submit_log_file=$SUBMIT_LOG_FILE"
while IFS= read -r storage_line; do
  log_line "$storage_line"
done < <(sycophancy_bias_print_storage_env "[submit-${JOB_DATE_TAG}]")

ENV_PYTHON_FOR_REGISTRY="${ENV_PYTHON_FOR_REGISTRY:-python}"
DEFAULT_BIAS_TYPES_CSV="$(PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$ENV_PYTHON_FOR_REGISTRY" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=False)))')"
BIAS_TYPES_CSV="${BIAS_TYPES_CSV:-$DEFAULT_BIAS_TYPES_CSV}"

TASK_LABELS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)
MODEL_IDS=(
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
)
DATASET_NAMES=(
  commonsense_qa
  commonsense_qa
  arc_challenge
  arc_challenge
)
RUN_SLUGS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)
REQUEST_STRENGTHS=(
  weak
  strong
)

TASK_FILTER="${TASK_FILTER:-}"
REQUEST_FILTER="${REQUEST_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_ANALYSIS_ONLY="${SUBMIT_ANALYSIS_ONLY:-0}"
SAMPLING_JOB_ID="${SAMPLING_JOB_ID:-}"
BASELINE_TAG="${BASELINE_TAG:-20260618}"
REQUEST_TAG="${REQUEST_TAG:-20260623}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-test}"

SELECTED_INDICES=()
for index in "${!TASK_LABELS[@]}"; do
  if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "${TASK_LABELS[$index]}" ]]; then
    SELECTED_INDICES+=("$index")
  fi
done
selected_task_count="${#SELECTED_INDICES[@]}"
if [[ "$selected_task_count" -eq 0 ]]; then
  printf '%s\n' "TASK_FILTER did not match a known task label: $TASK_FILTER" >&2
  exit 1
fi

SELECTED_REQUESTS=()
for request_strength in "${REQUEST_STRENGTHS[@]}"; do
  if [[ -z "$REQUEST_FILTER" || "$REQUEST_FILTER" == "$request_strength" ]]; then
    SELECTED_REQUESTS+=("$request_strength")
  fi
done
selected_request_count="${#SELECTED_REQUESTS[@]}"
if [[ "$selected_request_count" -eq 0 ]]; then
  printf '%s\n' "REQUEST_FILTER did not match weak or strong: $REQUEST_FILTER" >&2
  exit 1
fi

sampling_array_end=$((selected_task_count * selected_request_count - 1))
analysis_array_end="$sampling_array_end"
SAMPLING_ARRAY="${SAMPLING_ARRAY:-0-${sampling_array_end}}"
ANALYSIS_ARRAY="${ANALYSIS_ARRAY:-0-${analysis_array_end}}"
SAMPLING_TIME="${SAMPLING_TIME:-12:00:00}"
SAMPLING_MEM="${SAMPLING_MEM:-100G}"
ANALYSIS_TIME="${ANALYSIS_TIME:-12:00:00}"
ANALYSIS_MEM="${ANALYSIS_MEM:-100G}"
SUMMARY_TIME="${SUMMARY_TIME:-00:30:00}"
SUMMARY_MEM="${SUMMARY_MEM:-8G}"
SBATCH_CPUS="${SBATCH_CPUS:-2}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu,seas_gpu,gpu_h200}"
SUMMARY_CPUS="${SUMMARY_CPUS:-1}"
SUMMARY_PARTITION="${SUMMARY_PARTITION:-$SBATCH_PARTITION}"
SUMMARY_EMAIL_TO="${SUMMARY_EMAIL_TO:-itaishapira@g.harvard.edu}"
SUMMARY_EMAIL_SUBJECT="${SUMMARY_EMAIL_SUBJECT:-Anti-sycophancy request results: main mitigation stats}"
SEND_SUMMARY_EMAIL="${SEND_SUMMARY_EMAIL:-1}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$OUT_DIR/_comparisons/anti_sycophancy_request_20260623}"

export TASK_FILTER
export REQUEST_FILTER
export BIAS_TYPES_CSV
export BASELINE_TAG
export REQUEST_TAG
export ANALYSIS_SPLIT
export SUMMARY_EMAIL_TO
export SUMMARY_EMAIL_SUBJECT
export SEND_SUMMARY_EMAIL
export COMPARISON_ROOT
export OUT_DIR
export LOG_ROOT

printf 'array_task\ttask_label\tdataset_name\tmodel_id\trequest_strength\trun_name\trun_dir\n' > "$SAMPLING_TASK_MATRIX"
printf 'array_task\ttask_label\tdataset_name\tmodel_id\trequest_strength\trequest_run_dir\tbaseline_sampling_run_dir\tbaseline_random_all_probe_run_dir\tcomparison_output_dir\n' > "$ANALYSIS_TASK_MATRIX"
array_task=0
for base_index in "${SELECTED_INDICES[@]}"; do
  task_label="${TASK_LABELS[$base_index]}"
  dataset_name="${DATASET_NAMES[$base_index]}"
  model_id="${MODEL_IDS[$base_index]}"
  run_slug="${RUN_SLUGS[$base_index]}"
  model_slug="$(slugify "$model_id")"
  for request_strength in "${SELECTED_REQUESTS[@]}"; do
    request_run_name="${run_slug}_antisyc_${request_strength}_sampling_${REQUEST_TAG}"
    request_run_dir="$OUT_DIR/$model_slug/$dataset_name/$request_run_name"
    baseline_sampling_run_dir="$OUT_DIR/$model_slug/$dataset_name/${run_slug}_allfamilies_sampling_${BASELINE_TAG}"
    baseline_probe_run_dir="$OUT_DIR/$model_slug/$dataset_name/${run_slug}_allfamilies_probe_random_all_${BASELINE_TAG}"
    comparison_dataset_model="${dataset_name}_${model_slug}"
    comparison_output_dir="$OUT_DIR/_comparisons/anti_sycophancy_request_20260623/$comparison_dataset_model/$request_strength"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$array_task" "$task_label" "$dataset_name" "$model_id" "$request_strength" "$request_run_name" "$request_run_dir" >> "$SAMPLING_TASK_MATRIX"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$array_task" "$task_label" "$dataset_name" "$model_id" "$request_strength" \
      "$request_run_dir" "$baseline_sampling_run_dir" "$baseline_probe_run_dir" "$comparison_output_dir" >> "$ANALYSIS_TASK_MATRIX"
    array_task=$((array_task + 1))
  done
done

sampling_cmd=(
  sbatch
  --parsable
  --job-name "syco_antisyc_samp_${JOB_DATE_TAG}"
  --array "$SAMPLING_ARRAY"
  --time "$SAMPLING_TIME"
  --mem "$SAMPLING_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/${BUNDLE_NAME}/sampling_array.sbatch"
)

log_line "[submit-${JOB_DATE_TAG}] selected_task_count=$selected_task_count selected_request_count=$selected_request_count"
log_line "[submit-${JOB_DATE_TAG}] bias_types=$BIAS_TYPES_CSV"
log_line "[submit-${JOB_DATE_TAG}] baseline_tag=$BASELINE_TAG request_tag=$REQUEST_TAG split=$ANALYSIS_SPLIT"
log_line "[submit-${JOB_DATE_TAG}] sampling_task_matrix=$SAMPLING_TASK_MATRIX"
log_line "[submit-${JOB_DATE_TAG}] analysis_task_matrix=$ANALYSIS_TASK_MATRIX"
log_line "[submit-${JOB_DATE_TAG}] sampling_slurm_log_dir=$SAMPLING_SLURM_LOG_DIR"
log_line "[submit-${JOB_DATE_TAG}] analysis_slurm_log_dir=$ANALYSIS_SLURM_LOG_DIR"
log_line "[submit-${JOB_DATE_TAG}] summary_slurm_log_dir=$SUMMARY_SLURM_LOG_DIR"
log_line "[submit-${JOB_DATE_TAG}] structured_log_root=$STRUCTURED_LOG_ROOT"
log_line "[submit-${JOB_DATE_TAG}] comparison_root=$COMPARISON_ROOT"
log_line "[submit-${JOB_DATE_TAG}] summary_email_to=$SUMMARY_EMAIL_TO send_summary_email=$SEND_SUMMARY_EMAIL"

if [[ "$SUBMIT_ANALYSIS_ONLY" != "1" ]]; then
  log_cmd "[submit-${JOB_DATE_TAG}] sampling: " "${sampling_cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    sampling_job_id="${SAMPLING_JOB_ID:-<sampling_job_id>}"
  else
    sampling_job_id="$("${sampling_cmd[@]}")"
    log_line "[submit-${JOB_DATE_TAG}] sampling_job_id=$sampling_job_id"
  fi
else
  if [[ -z "$SAMPLING_JOB_ID" ]]; then
    printf '%s\n' "SUBMIT_ANALYSIS_ONLY=1 requires SAMPLING_JOB_ID for dependency context." >&2
    exit 1
  fi
  sampling_job_id="$SAMPLING_JOB_ID"
fi

analysis_cmd=(
  sbatch
  --parsable
  --job-name "syco_antisyc_eval_${JOB_DATE_TAG}"
  --dependency "afterok:${sampling_job_id}"
  --array "$ANALYSIS_ARRAY"
  --time "$ANALYSIS_TIME"
  --mem "$ANALYSIS_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$ANALYSIS_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$ANALYSIS_SLURM_LOG_DIR/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/${BUNDLE_NAME}/analysis_array.sbatch"
)

log_cmd "[submit-${JOB_DATE_TAG}] analysis: " "${analysis_cmd[@]}"

analysis_job_id="<analysis_job_id>"
if [[ "$DRY_RUN" != "1" ]]; then
  analysis_job_id="$("${analysis_cmd[@]}")"
  log_line "[submit-${JOB_DATE_TAG}] analysis_job_id=$analysis_job_id dependency=afterok:$sampling_job_id"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  sampling_job_id="${sampling_job_id//<sampling_job_id>/dryrun_sampling_job_id}"
  analysis_job_id="${analysis_job_id//<analysis_job_id>/dryrun_analysis_job_id}"
fi

summary_cmd=(
  sbatch
  --parsable
  --job-name "syco_antisyc_mail_${JOB_DATE_TAG}"
  --dependency "afterok:${analysis_job_id}"
  --time "$SUMMARY_TIME"
  --mem "$SUMMARY_MEM"
  --cpus-per-task "$SUMMARY_CPUS"
  --partition "$SUMMARY_PARTITION"
  --output "$SUMMARY_SLURM_LOG_DIR/%x.%j.out"
  --error "$SUMMARY_SLURM_LOG_DIR/%x.%j.err"
  "jobs/sycophancy_bias_probe/${BUNDLE_NAME}/summary_email.sbatch"
)

log_cmd "[submit-${JOB_DATE_TAG}] summary_email: " "${summary_cmd[@]}"

summary_job_id="<summary_job_id>"
if [[ "$DRY_RUN" != "1" ]]; then
  summary_job_id="$("${summary_cmd[@]}")"
  log_line "[submit-${JOB_DATE_TAG}] summary_job_id=$summary_job_id dependency=afterok:$analysis_job_id"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  summary_job_id="${summary_job_id//<summary_job_id>/dryrun_summary_job_id}"
fi

{
  printf 'BUNDLE_NAME=%q\n' "$BUNDLE_NAME"
  printf 'SUBMITTED_AT=%q\n' "$(iso_now)"
  printf 'ROOT_DIR=%q\n' "$ROOT_DIR"
  printf 'SYCOPHANCY_STORAGE_ROOT=%q\n' "$SYCOPHANCY_STORAGE_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'SLURM_LOG_ROOT=%q\n' "$SLURM_LOG_ROOT"
  printf 'SAMPLING_SLURM_LOG_DIR=%q\n' "$SAMPLING_SLURM_LOG_DIR"
  printf 'ANALYSIS_SLURM_LOG_DIR=%q\n' "$ANALYSIS_SLURM_LOG_DIR"
  printf 'SUMMARY_SLURM_LOG_DIR=%q\n' "$SUMMARY_SLURM_LOG_DIR"
  printf 'STRUCTURED_LOG_ROOT=%q\n' "$STRUCTURED_LOG_ROOT"
  printf 'SUBMIT_LOG_FILE=%q\n' "$SUBMIT_LOG_FILE"
  printf 'SAMPLING_TASK_MATRIX=%q\n' "$SAMPLING_TASK_MATRIX"
  printf 'ANALYSIS_TASK_MATRIX=%q\n' "$ANALYSIS_TASK_MATRIX"
  printf 'SAMPLING_JOB_ID=%q\n' "$sampling_job_id"
  printf 'ANALYSIS_JOB_ID=%q\n' "$analysis_job_id"
  printf 'SUMMARY_JOB_ID=%q\n' "$summary_job_id"
  printf 'TASK_FILTER=%q\n' "$TASK_FILTER"
  printf 'REQUEST_FILTER=%q\n' "$REQUEST_FILTER"
  printf 'BIAS_TYPES_CSV=%q\n' "$BIAS_TYPES_CSV"
  printf 'BASELINE_TAG=%q\n' "$BASELINE_TAG"
  printf 'REQUEST_TAG=%q\n' "$REQUEST_TAG"
  printf 'ANALYSIS_SPLIT=%q\n' "$ANALYSIS_SPLIT"
  printf 'SUMMARY_EMAIL_TO=%q\n' "$SUMMARY_EMAIL_TO"
  printf 'SUMMARY_EMAIL_SUBJECT=%q\n' "$SUMMARY_EMAIL_SUBJECT"
  printf 'SEND_SUMMARY_EMAIL=%q\n' "$SEND_SUMMARY_EMAIL"
  printf 'COMPARISON_ROOT=%q\n' "$COMPARISON_ROOT"
  printf 'OUT_DIR=%q\n' "$OUT_DIR"
  printf 'HF_HUB_CACHE=%q\n' "$HF_HUB_CACHE"
  printf 'HUGGINGFACE_HUB_CACHE=%q\n' "$HUGGINGFACE_HUB_CACHE"
  printf 'TRANSFORMERS_CACHE=%q\n' "$TRANSFORMERS_CACHE"
  printf 'HF_DATASETS_CACHE=%q\n' "$HF_DATASETS_CACHE"
  printf 'HF_HOME=%q\n' "$HF_HOME"
  printf 'TRITON_CACHE_DIR=%q\n' "$TRITON_CACHE_DIR"
  printf 'WANDB_DIR=%q\n' "$WANDB_DIR"
  printf 'WANDB_CACHE_DIR=%q\n' "$WANDB_CACHE_DIR"
  printf 'WANDB_CONFIG_DIR=%q\n' "$WANDB_CONFIG_DIR"
  printf 'WANDB_DATA_DIR=%q\n' "$WANDB_DATA_DIR"
  printf 'TMPDIR=%q\n' "$TMPDIR"
  printf 'MPLCONFIGDIR=%q\n' "$MPLCONFIGDIR"
  printf 'TORCH_HOME=%q\n' "$TORCH_HOME"
  printf 'XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
  printf 'DRY_RUN=%q\n' "$DRY_RUN"
} > "$SUBMISSION_ENV_FILE"
cp "$SUBMISSION_ENV_FILE" "$LATEST_SUBMISSION_ENV_FILE"
log_line "[submit-${JOB_DATE_TAG}] submission_env=$SUBMISSION_ENV_FILE"
log_line "[submit-${JOB_DATE_TAG}] latest_submission_env=$LATEST_SUBMISSION_ENV_FILE"

log_line "[submit-${JOB_DATE_TAG}] next_status_command=bash jobs/sycophancy_bias_probe/${BUNDLE_NAME}/status_anti_sycophancy_request_sharded_20260623.sh"
log_line "[submit-${JOB_DATE_TAG}] structured_logs_root=$STRUCTURED_LOG_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
