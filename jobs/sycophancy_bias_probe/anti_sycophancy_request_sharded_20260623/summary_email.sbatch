#!/bin/bash
#SBATCH --job-name=syco_antisyc_mail_20260623
#SBATCH --partition=sapphire,shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/summary/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/summary/%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

BUNDLE_NAME="anti_sycophancy_request_sharded_20260623"
SCRIPT_TAG="anti-sycophancy-summary-email-20260623"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

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

mkdir -p \
  "$OUT_DIR" \
  "$LOG_ROOT" \
  "$LOG_ROOT/slurm/summary" \
  "$LOG_ROOT/by_task/summary" \
  "$LOG_ROOT/summary"

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

send_email() {
  local recipient="$1"
  local subject="$2"
  local body_path="$3"
  if command -v mail >/dev/null 2>&1; then
    mail -s "$subject" "$recipient" < "$body_path"
    return 0
  fi
  if command -v mailx >/dev/null 2>&1; then
    mailx -s "$subject" "$recipient" < "$body_path"
    return 0
  fi
  if command -v sendmail >/dev/null 2>&1; then
    {
      printf 'To: %s\n' "$recipient"
      printf 'Subject: %s\n' "$subject"
      printf '\n'
      cat "$body_path"
    } | sendmail -t
    return 0
  fi
  printf '%s\n' "No mail, mailx, or sendmail command found; summary written but not emailed." >&2
  return 1
}

SUMMARY_EMAIL_TO="${SUMMARY_EMAIL_TO:-itaishapira@g.harvard.edu}"
SUMMARY_EMAIL_SUBJECT="${SUMMARY_EMAIL_SUBJECT:-Anti-sycophancy request results: main mitigation stats}"
SEND_SUMMARY_EMAIL="${SEND_SUMMARY_EMAIL:-1}"
COMPARISON_ROOT="${COMPARISON_ROOT:-$OUT_DIR/_comparisons/anti_sycophancy_request_20260623}"
SUMMARY_STEM="$(date +%Y%m%dT%H%M%S%z)_job_${SLURM_JOB_ID:-local}"
SUMMARY_TXT="$LOG_ROOT/summary/anti_sycophancy_request_summary_${SUMMARY_STEM}.txt"
SUMMARY_CSV="$LOG_ROOT/summary/anti_sycophancy_request_summary_${SUMMARY_STEM}.csv"
DETAIL_CSV="$LOG_ROOT/summary/anti_sycophancy_request_detail_${SUMMARY_STEM}.csv"

STRUCTURED_LOG_DIR="$LOG_ROOT/by_task/summary/job_${SLURM_JOB_ID:-local}"
mkdir -p "$STRUCTURED_LOG_DIR"
STRUCTURED_STDOUT="$STRUCTURED_LOG_DIR/task_summary.out"
STRUCTURED_STDERR="$STRUCTURED_LOG_DIR/task_summary.err"
exec > >(tee -a "$STRUCTURED_STDOUT") 2> >(tee -a "$STRUCTURED_STDERR" >&2)

printf '[%s] started_at=%s hostname=%s pwd=%s\n' "$SCRIPT_TAG" "$(iso_now)" "$(hostname)" "$PWD"
printf '[%s] comparison_root=%s\n' "$SCRIPT_TAG" "$COMPARISON_ROOT"
printf '[%s] summary_txt=%s\n' "$SCRIPT_TAG" "$SUMMARY_TXT"
printf '[%s] summary_csv=%s\n' "$SCRIPT_TAG" "$SUMMARY_CSV"
printf '[%s] detail_csv=%s\n' "$SCRIPT_TAG" "$DETAIL_CSV"
printf '[%s] summary_email_to=%s send_summary_email=%s\n' "$SCRIPT_TAG" "$SUMMARY_EMAIL_TO" "$SEND_SUMMARY_EMAIL"
sycophancy_bias_print_storage_env "[$SCRIPT_TAG]"

cmd=(
  "$ENV_PYTHON" scripts/summarize_anti_sycophancy_request.py
  --comparison_root "$COMPARISON_ROOT"
  --output_txt "$SUMMARY_TXT"
  --output_csv "$SUMMARY_CSV"
  --detail_csv "$DETAIL_CSV"
)
printf '[%s] command: ' "$SCRIPT_TAG"
printf '%q ' "${cmd[@]}"
printf '\n'

START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e

if [[ "$STATUS" -eq 0 ]]; then
  printf '[%s] summary preview:\n' "$SCRIPT_TAG"
  sed -n '1,80p' "$SUMMARY_TXT"
  if [[ "$SEND_SUMMARY_EMAIL" == "1" ]]; then
    printf '[%s] sending summary email...\n' "$SCRIPT_TAG"
    set +e
    send_email "$SUMMARY_EMAIL_TO" "$SUMMARY_EMAIL_SUBJECT" "$SUMMARY_TXT"
    MAIL_STATUS="$?"
    set -e
    if [[ "$MAIL_STATUS" -ne 0 ]]; then
      STATUS="$MAIL_STATUS"
    fi
  fi
fi

END_EPOCH="$(date +%s)"
printf '[%s] finished_at=%s exit_status=%s elapsed_seconds=%s\n' \
  "$SCRIPT_TAG" "$(iso_now)" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
