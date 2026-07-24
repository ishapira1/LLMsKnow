#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
REPO_DIR="$ROOT_DIR"
export REPO_DIR

source jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/common.sh

SUBMISSION_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
export SUBMISSION_GIT_COMMIT
DRY_RUN="${DRY_RUN:-0}"

SUBMISSION_STEM="$(date +%Y%m%dT%H%M%S%z)_pid_$$"
SUBMIT_LOG="$LOG_ROOT/submit/submit_$SUBMISSION_STEM.log"
SUBMISSION_ENV="$LOG_ROOT/submit/submission_$SUBMISSION_STEM.env"
LATEST_ENV="$LOG_ROOT/submit/latest_submission.env"

log_command() {
  local label="$1"
  shift
  printf '%s' "$label" | tee -a "$SUBMIT_LOG"
  printf '%q ' "$@" | tee -a "$SUBMIT_LOG"
  printf '\n' | tee -a "$SUBMIT_LOG"
}

IFS=',' read -r -a LAYERS <<< "$PILOT_LAYERS_CSV"
if (( ${#LAYERS[@]} < 5 )); then
  printf '%s\n' "This overnight pilot requires at least five prespecified layers." >&2
  exit 1
fi
if [[ "$PILOT_MAX_QUESTIONS" -lt 100 ]]; then
  printf '%s\n' "This overnight pilot requires at least 100 validation questions." >&2
  exit 1
fi

test -d "$SOURCE_RUN_DIR"
test -s "$DIRECTIONS_PATH"
test -s "$(dirname "$DIRECTIONS_PATH")/manifest.json"

runtime_cmd=(
  "$ENV_PYTHON" "$RUNTIME_CONTRACT_PATH"
  --hf-cache-dir "$HF_HUB_CACHE"
  --local-files-only
  --model-config Qwen/Qwen2.5-7B-Instruct
)
log_command "[preflight] runtime: " "${runtime_cmd[@]}"
"${runtime_cmd[@]}" 2>&1 | tee -a "$SUBMIT_LOG"

PREFLIGHT_DIR="$LOG_ROOT/submit/preflight_$SUBMISSION_STEM"
mkdir -p "$PREFLIGHT_DIR"
validate_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" validate-source
  --source-run-dir "$SOURCE_RUN_DIR"
  --output-dir "$PREFLIGHT_DIR/source"
)
log_command "[preflight] source: " "${validate_cmd[@]}"
"${validate_cmd[@]}" 2>&1 | tee -a "$SUBMIT_LOG"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=int(d.get("pairs_by_split",{}).get("val",0)); assert n >= int(sys.argv[2]), (n, sys.argv[2])' \
  "$PREFLIGHT_DIR/source/source_validation.json" "$PILOT_MAX_QUESTIONS"

if [[ "$DRY_RUN" == "1" ]]; then
  PREPARE_ROOT="$PREFLIGHT_DIR/prepared_output"
else
  if find "$CELL_ROOT/layers" -type f \( -name 'item_results_*.jsonl' -o -name 'manifest_*.json' \) -print -quit 2>/dev/null | grep -q .; then
    printf '%s\n' "Refusing to reuse a pilot output root containing result shards: $CELL_ROOT" >&2
    exit 1
  fi
  if [[ -e "$CELL_ROOT/selected_patch_layers.json" ]]; then
    printf '%s\n' "Refusing to overwrite an existing pilot layer plan: $CELL_ROOT/selected_patch_layers.json" >&2
    exit 1
  fi
  PREPARE_ROOT="$CELL_ROOT"
fi

prepare_cmd=(
  "$ENV_PYTHON"
  "$REPO_DIR/jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/summarize_pilot.py"
  prepare
  --output-root "$PREPARE_ROOT"
  --source-run-dir "$SOURCE_RUN_DIR"
  --directions-path "$DIRECTIONS_PATH"
  --layers "$PILOT_LAYERS_CSV"
  --max-questions "$PILOT_MAX_QUESTIONS"
)
log_command "[preflight] frozen pilot plan: " "${prepare_cmd[@]}"
"${prepare_cmd[@]}" 2>&1 | tee -a "$SUBMIT_LOG"

array_spec="0-$(( ${#LAYERS[@]} - 1 ))%${#LAYERS[@]}"
GPU_PARTITION="${GPU_PARTITION:-gpu_requeue}"
GPU_GRES="${GPU_GRES:-gpu:nvidia_h200:1}"
GPU_TIME="${GPU_TIME:-10:00:00}"
GPU_MEM="${GPU_MEM:-64G}"

dose_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name syco_qwen_md_pilot_20260724
  --partition "$GPU_PARTITION"
  --gres "$GPU_GRES"
  --array "$array_spec"
  --cpus-per-task 4
  --mem "$GPU_MEM"
  --time "$GPU_TIME"
  --requeue
  --output "$LOG_ROOT/slurm/dose/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/dose/%x.%A_%a.err"
  jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/dose_array.sbatch
)
log_command "[submit] dose array: " "${dose_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  DOSE_JOB_ID="dryrun_dose_job_id"
else
  DOSE_JOB_ID="$("${dose_cmd[@]}")"
  DOSE_JOB_ID="${DOSE_JOB_ID%%;*}"
fi

aggregate_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name syco_qwen_md_agg_20260724
  --dependency "afterok:$DOSE_JOB_ID"
  --partition shared,sapphire
  --cpus-per-task 4
  --mem 64G
  --time 03:00:00
  --output "$LOG_ROOT/slurm/aggregate/%x.%j.out"
  --error "$LOG_ROOT/slurm/aggregate/%x.%j.err"
  jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/aggregate.sbatch
)
log_command "[submit] aggregate: " "${aggregate_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  AGGREGATE_JOB_ID="dryrun_aggregate_job_id"
else
  AGGREGATE_JOB_ID="$("${aggregate_cmd[@]}")"
  AGGREGATE_JOB_ID="${AGGREGATE_JOB_ID%%;*}"
fi

{
  printf 'BUNDLE_NAME=%q\n' "$BUNDLE_NAME"
  printf 'SUBMITTED_AT=%q\n' "$(iso_now)"
  printf 'SUBMISSION_GIT_COMMIT=%q\n' "$SUBMISSION_GIT_COMMIT"
  printf 'REPO_DIR=%q\n' "$REPO_DIR"
  printf 'ENV_PYTHON=%q\n' "$ENV_PYTHON"
  printf 'PILOT_RUN_ID=%q\n' "$PILOT_RUN_ID"
  printf 'PILOT_LAYERS_CSV=%q\n' "$PILOT_LAYERS_CSV"
  printf 'PILOT_MAX_QUESTIONS=%q\n' "$PILOT_MAX_QUESTIONS"
  printf 'PILOT_ALPHAS=%q\n' "$PILOT_ALPHAS"
  printf 'PILOT_CONTROL_SEEDS=%q\n' "$PILOT_CONTROL_SEEDS"
  printf 'SOURCE_RUN_DIR=%q\n' "$SOURCE_RUN_DIR"
  printf 'DIRECTIONS_PATH=%q\n' "$DIRECTIONS_PATH"
  printf 'CELL_ROOT=%q\n' "$CELL_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'GPU_PARTITION=%q\n' "$GPU_PARTITION"
  printf 'GPU_GRES=%q\n' "$GPU_GRES"
  printf 'DOSE_JOB_ID=%q\n' "$DOSE_JOB_ID"
  printf 'AGGREGATE_JOB_ID=%q\n' "$AGGREGATE_JOB_ID"
} | tee "$SUBMISSION_ENV"
cp "$SUBMISSION_ENV" "$LATEST_ENV"

printf '[submit] dry_run=%s dose_job_id=%s aggregate_job_id=%s manifest=%s\n' \
  "$DRY_RUN" "$DOSE_JOB_ID" "$AGGREGATE_JOB_ID" "$SUBMISSION_ENV"

