#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

LOG_ROOT="jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260616"
SLURM_LOG_ROOT="$LOG_ROOT/slurm"
SAMPLING_SLURM_LOG_DIR="$SLURM_LOG_ROOT/sampling"
PROBE_SLURM_LOG_DIR="$SLURM_LOG_ROOT/probes"
STRUCTURED_LOG_ROOT="$LOG_ROOT/by_task"
SUBMIT_LOG_DIR="$LOG_ROOT/submit"
mkdir -p "$SAMPLING_SLURM_LOG_DIR" "$PROBE_SLURM_LOG_DIR" "$STRUCTURED_LOG_ROOT" "$SUBMIT_LOG_DIR"

SUBMIT_LOG_FILE="$SUBMIT_LOG_DIR/submit_$(date +%Y%m%dT%H%M%S%z)_pid_$$.log"

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

log_line "[submit-sharded-20260616] submit_log_file=$SUBMIT_LOG_FILE"

ENV_PYTHON_FOR_REGISTRY="${ENV_PYTHON_FOR_REGISTRY:-python}"
DEFAULT_PROBE_FAMILIES_CSV="$(PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$ENV_PYTHON_FOR_REGISTRY" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=True)))')"
PROBE_FAMILIES_CSV="${PROBE_FAMILIES_CSV:-$DEFAULT_PROBE_FAMILIES_CSV}"

TASK_LABELS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)

TASK_FILTER="${TASK_FILTER:-}"
PROBE_FAMILY_FILTER="${PROBE_FAMILY_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_PROBES_ONLY="${SUBMIT_PROBES_ONLY:-0}"
SAMPLING_JOB_ID="${SAMPLING_JOB_ID:-}"

selected_task_count=0
for label in "${TASK_LABELS[@]}"; do
  if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "$label" ]]; then
    selected_task_count=$((selected_task_count + 1))
  fi
done
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

sampling_cmd=(
  sbatch
  --parsable
  --job-name syco_allfam_sample_20260616
  --array "$SAMPLING_ARRAY"
  --time "$SAMPLING_TIME"
  --mem "$SAMPLING_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$SAMPLING_SLURM_LOG_DIR/%x.%A_%a.err"
  jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/sampling_array.sbatch
)

log_line "[submit-sharded-20260616] selected_task_count=$selected_task_count probe_family_count=$probe_family_count"
log_line "[submit-sharded-20260616] probe_families=$PROBE_FAMILIES_CSV"
log_line "[submit-sharded-20260616] sampling_slurm_log_dir=$SAMPLING_SLURM_LOG_DIR"
log_line "[submit-sharded-20260616] probe_slurm_log_dir=$PROBE_SLURM_LOG_DIR"
log_line "[submit-sharded-20260616] structured_log_root=$STRUCTURED_LOG_ROOT"

if [[ "$SUBMIT_PROBES_ONLY" != "1" ]]; then
  log_cmd "[submit-sharded-20260616] sampling: " "${sampling_cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    sampling_job_id="${SAMPLING_JOB_ID:-<sampling_job_id>}"
  else
    sampling_job_id="$("${sampling_cmd[@]}")"
    log_line "[submit-sharded-20260616] sampling_job_id=$sampling_job_id"
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
  --job-name syco_allfam_probe_20260616
  --dependency "afterok:${sampling_job_id}"
  --array "$PROBE_ARRAY"
  --time "$PROBE_TIME"
  --mem "$PROBE_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  --output "$PROBE_SLURM_LOG_DIR/%x.%A_%a.out"
  --error "$PROBE_SLURM_LOG_DIR/%x.%A_%a.err"
  jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/probe_family_array.sbatch
)

log_cmd "[submit-sharded-20260616] probes: " "${probe_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

probe_job_id="$("${probe_cmd[@]}")"
log_line "[submit-sharded-20260616] probe_job_id=$probe_job_id dependency=afterok:$sampling_job_id"
