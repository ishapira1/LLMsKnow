#!/bin/bash
set -euo pipefail

BUNDLE_NAME="full_allfamilies_paraphrase_sharded_20260618"
JOB_DATE_TAG="20260618"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $ROOT_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"

SLURM_LOG_ROOT="$LOG_ROOT/slurm"
SAMPLING_SLURM_LOG_DIR="$SLURM_LOG_ROOT/sampling"
PROBE_SLURM_LOG_DIR="$SLURM_LOG_ROOT/probes"
STRUCTURED_LOG_ROOT="$LOG_ROOT/by_task"
SUBMIT_LOG_DIR="$LOG_ROOT/submit"
mkdir -p "$SAMPLING_SLURM_LOG_DIR" "$PROBE_SLURM_LOG_DIR" "$STRUCTURED_LOG_ROOT" "$SUBMIT_LOG_DIR"

SUBMISSION_STEM="$(date +%Y%m%dT%H%M%S%z)_pid_$$"
SUBMIT_LOG_FILE="$SUBMIT_LOG_DIR/submit_${SUBMISSION_STEM}.log"
SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/submission_${SUBMISSION_STEM}.env"
LATEST_SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/latest_submission.env"
SAMPLING_TASK_MATRIX="$SUBMIT_LOG_DIR/sampling_task_matrix_${SUBMISSION_STEM}.tsv"
PROBE_TASK_MATRIX="$SUBMIT_LOG_DIR/probe_task_matrix_${SUBMISSION_STEM}.tsv"

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
DEFAULT_PROBE_FAMILIES_CSV="$(PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$ENV_PYTHON_FOR_REGISTRY" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=True)))')"
PROBE_FAMILIES_CSV="${PROBE_FAMILIES_CSV:-$DEFAULT_PROBE_FAMILIES_CSV}"

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

TASK_FILTER="${TASK_FILTER:-}"
PROBE_FAMILY_FILTER="${PROBE_FAMILY_FILTER:-}"
PARAPHRASE_ARTIFACT_PATH="${PARAPHRASE_ARTIFACT_PATH:-data/ad_hoc/paraphrase_robustness_test_stems_v1}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_PROBES_ONLY="${SUBMIT_PROBES_ONLY:-0}"
SAMPLING_JOB_ID="${SAMPLING_JOB_ID:-}"

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

IFS=, read -r -a requested_probe_families <<< "$PROBE_FAMILIES_CSV"
if [[ -n "$PROBE_FAMILY_FILTER" ]]; then
  found_probe_family=0
  for family in "${requested_probe_families[@]}"; do
    if [[ "$family" == "$PROBE_FAMILY_FILTER" ]]; then
      found_probe_family=1
      break
    fi
  done
  if [[ "$found_probe_family" -ne 1 ]]; then
    printf '%s\n' "PROBE_FAMILY_FILTER did not match a trainable probe family: $PROBE_FAMILY_FILTER" >&2
    printf '%s\n' "Valid probe families: $PROBE_FAMILIES_CSV" >&2
    exit 1
  fi
  PROBE_FAMILIES_CSV="$PROBE_FAMILY_FILTER"
  requested_probe_families=("$PROBE_FAMILY_FILTER")
fi

probe_family_count="${#requested_probe_families[@]}"
if [[ "$probe_family_count" -eq 0 ]]; then
  printf '%s\n' "No probe families selected." >&2
  exit 1
fi

sampling_array_end=$((selected_task_count - 1))
probe_array_end=$((selected_task_count * probe_family_count - 1))
SAMPLING_ARRAY="${SAMPLING_ARRAY:-0-${sampling_array_end}}"
PROBE_ARRAY="${PROBE_ARRAY:-0-${probe_array_end}}"
SAMPLING_TIME="${SAMPLING_TIME:-12:00:00}"
SAMPLING_MEM="${SAMPLING_MEM:-100G}"
PROBE_TIME="${PROBE_TIME:-24:00:00}"
PROBE_MEM="${PROBE_MEM:-100G}"
SBATCH_CPUS="${SBATCH_CPUS:-2}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu,seas_gpu,gpu_h200}"

export TASK_FILTER
export PROBE_FAMILY_FILTER
export PROBE_FAMILIES_CSV
export PARAPHRASE_ARTIFACT_PATH
export OUT_DIR
export LOG_ROOT

printf 'array_task\ttask_label\tdataset_name\tmodel_id\trun_name\trun_dir\n' > "$SAMPLING_TASK_MATRIX"
sampling_task_index=0
for base_index in "${SELECTED_INDICES[@]}"; do
  task_label="${TASK_LABELS[$base_index]}"
  dataset_name="${DATASET_NAMES[$base_index]}"
  model_id="${MODEL_IDS[$base_index]}"
  run_name="${RUN_SLUGS[$base_index]}_allfamilies_sampling_${JOB_DATE_TAG}"
  model_slug="$(slugify "$model_id")"
  run_dir="$OUT_DIR/$model_slug/$dataset_name/$run_name"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sampling_task_index" "$task_label" "$dataset_name" "$model_id" "$run_name" "$run_dir" >> "$SAMPLING_TASK_MATRIX"
  sampling_task_index=$((sampling_task_index + 1))
done

printf 'array_task\ttask_label\tprobe_family\tdataset_name\tmodel_id\trun_name\trun_dir\n' > "$PROBE_TASK_MATRIX"
probe_task_index=0
for base_index in "${SELECTED_INDICES[@]}"; do
  task_label="${TASK_LABELS[$base_index]}"
  dataset_name="${DATASET_NAMES[$base_index]}"
  model_id="${MODEL_IDS[$base_index]}"
  model_slug="$(slugify "$model_id")"
  for family in "${requested_probe_families[@]}"; do
    run_name="${RUN_SLUGS[$base_index]}_allfamilies_probe_${family}_${JOB_DATE_TAG}"
    run_dir="$OUT_DIR/$model_slug/$dataset_name/$run_name"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$probe_task_index" "$task_label" "$family" "$dataset_name" "$model_id" "$run_name" "$run_dir" >> "$PROBE_TASK_MATRIX"
    probe_task_index=$((probe_task_index + 1))
  done
done

sampling_cmd=(
  sbatch
  --parsable
  --job-name "syco_allfam_samp_${JOB_DATE_TAG}"
  --array "$SAMPLING_ARRAY"
  --time "$SAMPLING_TIME"
  --mem "$SAMPLING_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/${BUNDLE_NAME}/sampling_array.sbatch"
)

log_line "[submit-${JOB_DATE_TAG}] selected_task_count=$selected_task_count probe_family_count=$probe_family_count"
log_line "[submit-${JOB_DATE_TAG}] probe_families=$PROBE_FAMILIES_CSV"
log_line "[submit-${JOB_DATE_TAG}] paraphrase_artifact_path=$PARAPHRASE_ARTIFACT_PATH"
log_line "[submit-${JOB_DATE_TAG}] sampling_task_matrix=$SAMPLING_TASK_MATRIX"
log_line "[submit-${JOB_DATE_TAG}] probe_task_matrix=$PROBE_TASK_MATRIX"
log_line "[submit-${JOB_DATE_TAG}] sampling_slurm_log_dir=$SAMPLING_SLURM_LOG_DIR"
log_line "[submit-${JOB_DATE_TAG}] probe_slurm_log_dir=$PROBE_SLURM_LOG_DIR"
log_line "[submit-${JOB_DATE_TAG}] structured_log_root=$STRUCTURED_LOG_ROOT"

if [[ "$SUBMIT_PROBES_ONLY" != "1" ]]; then
  log_cmd "[submit-${JOB_DATE_TAG}] sampling: " "${sampling_cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    sampling_job_id="${SAMPLING_JOB_ID:-<sampling_job_id>}"
  else
    sampling_job_id="$("${sampling_cmd[@]}")"
    log_line "[submit-${JOB_DATE_TAG}] sampling_job_id=$sampling_job_id"
  fi
else
  if [[ -z "$SAMPLING_JOB_ID" ]]; then
    printf '%s\n' "SUBMIT_PROBES_ONLY=1 requires SAMPLING_JOB_ID for dependency context." >&2
    exit 1
  fi
  sampling_job_id="$SAMPLING_JOB_ID"
fi

probe_cmd=(
  sbatch
  --parsable
  --job-name "syco_allfam_probe_${JOB_DATE_TAG}"
  --dependency "afterok:${sampling_job_id}"
  --array "$PROBE_ARRAY"
  --time "$PROBE_TIME"
  --mem "$PROBE_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$PROBE_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$PROBE_SLURM_LOG_DIR/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/${BUNDLE_NAME}/probe_family_array.sbatch"
)

log_cmd "[submit-${JOB_DATE_TAG}] probes: " "${probe_cmd[@]}"

probe_job_id="<probe_job_id>"
if [[ "$DRY_RUN" != "1" ]]; then
  probe_job_id="$("${probe_cmd[@]}")"
  log_line "[submit-${JOB_DATE_TAG}] probe_job_id=$probe_job_id dependency=afterok:$sampling_job_id"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  sampling_job_id="${sampling_job_id//<sampling_job_id>/dryrun_sampling_job_id}"
  probe_job_id="${probe_job_id//<probe_job_id>/dryrun_probe_job_id}"
fi

{
  printf 'BUNDLE_NAME=%q\n' "$BUNDLE_NAME"
  printf 'SUBMITTED_AT=%q\n' "$(iso_now)"
  printf 'ROOT_DIR=%q\n' "$ROOT_DIR"
  printf 'SYCOPHANCY_STORAGE_ROOT=%q\n' "$SYCOPHANCY_STORAGE_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'SLURM_LOG_ROOT=%q\n' "$SLURM_LOG_ROOT"
  printf 'SAMPLING_SLURM_LOG_DIR=%q\n' "$SAMPLING_SLURM_LOG_DIR"
  printf 'PROBE_SLURM_LOG_DIR=%q\n' "$PROBE_SLURM_LOG_DIR"
  printf 'STRUCTURED_LOG_ROOT=%q\n' "$STRUCTURED_LOG_ROOT"
  printf 'SUBMIT_LOG_FILE=%q\n' "$SUBMIT_LOG_FILE"
  printf 'SAMPLING_TASK_MATRIX=%q\n' "$SAMPLING_TASK_MATRIX"
  printf 'PROBE_TASK_MATRIX=%q\n' "$PROBE_TASK_MATRIX"
  printf 'SAMPLING_JOB_ID=%q\n' "$sampling_job_id"
  printf 'PROBE_JOB_ID=%q\n' "$probe_job_id"
  printf 'TASK_FILTER=%q\n' "$TASK_FILTER"
  printf 'PROBE_FAMILY_FILTER=%q\n' "$PROBE_FAMILY_FILTER"
  printf 'PROBE_FAMILIES_CSV=%q\n' "$PROBE_FAMILIES_CSV"
  printf 'PARAPHRASE_ARTIFACT_PATH=%q\n' "$PARAPHRASE_ARTIFACT_PATH"
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

log_line "[submit-${JOB_DATE_TAG}] next_status_command=bash jobs/sycophancy_bias_probe/${BUNDLE_NAME}/status_full_allfamilies_paraphrase_sharded_20260618.sh"
log_line "[submit-${JOB_DATE_TAG}] structured_logs_root=$STRUCTURED_LOG_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
