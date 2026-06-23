#!/bin/bash
set -euo pipefail

BUNDLE_NAME="anti_sycophancy_request_sharded_20260623"
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

LATEST_SUBMISSION_ENV_FILE="$LOG_ROOT/submit/latest_submission.env"
if [[ ! -f "$LATEST_SUBMISSION_ENV_FILE" ]]; then
  printf '%s\n' "No latest submission env found: $LATEST_SUBMISSION_ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$LATEST_SUBMISSION_ENV_FILE"

printf '[status] bundle=%s\n' "$BUNDLE_NAME"
printf '[status] submitted_at=%s\n' "${SUBMITTED_AT:-unknown}"
printf '[status] sampling_job_id=%s\n' "${SAMPLING_JOB_ID:-unknown}"
printf '[status] analysis_job_id=%s\n' "${ANALYSIS_JOB_ID:-unknown}"
printf '[status] summary_job_id=%s\n' "${SUMMARY_JOB_ID:-unknown}"
printf '[status] log_root=%s\n' "${LOG_ROOT:-unknown}"
printf '[status] sampling_task_matrix=%s\n' "${SAMPLING_TASK_MATRIX:-unknown}"
printf '[status] analysis_task_matrix=%s\n' "${ANALYSIS_TASK_MATRIX:-unknown}"
printf '[status] structured_log_root=%s\n' "${STRUCTURED_LOG_ROOT:-unknown}"

if [[ -f "${SAMPLING_TASK_MATRIX:-}" ]]; then
  printf '\n[status] sampling task matrix:\n'
  sed -n '1,20p' "$SAMPLING_TASK_MATRIX"
fi
if [[ -f "${ANALYSIS_TASK_MATRIX:-}" ]]; then
  printf '\n[status] analysis task matrix:\n'
  sed -n '1,20p' "$ANALYSIS_TASK_MATRIX"
fi

if command -v squeue >/dev/null 2>&1; then
  job_filter=()
  if [[ -n "${SAMPLING_JOB_ID:-}" && "$SAMPLING_JOB_ID" != "dryrun_sampling_job_id" ]]; then
    job_filter+=("$SAMPLING_JOB_ID")
  fi
  if [[ -n "${ANALYSIS_JOB_ID:-}" && "$ANALYSIS_JOB_ID" != "dryrun_analysis_job_id" ]]; then
    job_filter+=("$ANALYSIS_JOB_ID")
  fi
  if [[ -n "${SUMMARY_JOB_ID:-}" && "$SUMMARY_JOB_ID" != "dryrun_summary_job_id" ]]; then
    job_filter+=("$SUMMARY_JOB_ID")
  fi
  if [[ "${#job_filter[@]}" -gt 0 ]]; then
    IFS=, eval 'jobs_csv="${job_filter[*]}"'
    printf '\n[status] squeue:\n'
    squeue -j "$jobs_csv" || true
  fi
fi
