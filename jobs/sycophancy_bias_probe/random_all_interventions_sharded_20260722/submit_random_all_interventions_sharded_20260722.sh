#!/bin/bash

set -euo pipefail

BUNDLE_NAME="random_all_interventions_sharded_20260722"
JOB_DATE_TAG="20260722"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
CALLER_ENV_PYTHON="${ENV_PYTHON:-}"

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $ROOT_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

# Bind every submitted task to the exact checkout containing this submitter.
# This intentionally overrides a stale REPO_DIR inherited from another checkout.
REPO_DIR="$ROOT_DIR"
export REPO_DIR
if [[ ! -f "$REPO_DIR/run_random_all_intervention.py" ]]; then
  printf '%s\n' "Missing intervention entrypoint in submitting checkout: $REPO_DIR/run_random_all_intervention.py" >&2
  exit 1
fi
SUBMISSION_GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$SUBMISSION_GIT_COMMIT" ]]; then
  printf '%s\n' "Unable to resolve the Git commit for submitting checkout: $REPO_DIR" >&2
  exit 1
fi

if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
  unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HOME
  unset TRITON_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  unset TMPDIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME OUT_DIR LOG_ROOT
fi

source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"

RUNTIME_ENV_DIR="${RANDOM_ALL_INTERVENTION_ENV_DIR:-$SYCOPHANCY_STORAGE_ROOT/python_envs/llmsknow_py310_torch220_transformers4423}"
ENV_PYTHON="${CALLER_ENV_PYTHON:-$RUNTIME_ENV_DIR/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' \
    "Missing validated intervention Python: $ENV_PYTHON. Run jobs/sycophancy_bias_probe/$BUNDLE_NAME/create_runtime_env.sh or set ENV_PYTHON explicitly." >&2
  exit 1
fi
RUNTIME_CONTRACT_PATH="$REPO_DIR/jobs/sycophancy_bias_probe/$BUNDLE_NAME/runtime_contract.py"
"$ENV_PYTHON" "$RUNTIME_CONTRACT_PATH"

SUBMIT_LOG_DIR="$LOG_ROOT/submit"
SUBMISSION_STEM="$(date +%Y%m%dT%H%M%S%z)_pid_$$"
SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-$SYCOPHANCY_BIAS_RESULTS_DIR}"
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-run_$SUBMISSION_STEM}"
INTERVENTION_BASE_ROOT="${INTERVENTION_BASE_ROOT:-$SYCOPHANCY_STORAGE_ROOT/LLMsKnow_results/sycophancy_bias_intervention/random_all_interventions_20260722}"
INTERVENTION_ROOT="${INTERVENTION_ROOT:-$INTERVENTION_BASE_ROOT/$EXPERIMENT_RUN_ID}"
sycophancy_bias_reject_home_path "INTERVENTION_ROOT" "$INTERVENTION_ROOT"
mkdir -p \
  "$SUBMIT_LOG_DIR" \
  "$LOG_ROOT/slurm/fit" \
  "$LOG_ROOT/slurm/localize" \
  "$LOG_ROOT/slurm/select_layers" \
  "$LOG_ROOT/slurm/dose_tune" \
  "$LOG_ROOT/slurm/select_dose" \
  "$LOG_ROOT/slurm/confirm" \
  "$LOG_ROOT/slurm/aggregate" \
  "$LOG_ROOT/by_task" \
  "$INTERVENTION_ROOT"

SUBMIT_LOG_FILE="$SUBMIT_LOG_DIR/submit_${SUBMISSION_STEM}.log"
SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/submission_${SUBMISSION_STEM}.env"
LATEST_SUBMISSION_ENV_FILE="$SUBMIT_LOG_DIR/latest_submission.env"
TASK_MATRIX="$SUBMIT_LOG_DIR/task_matrix_${SUBMISSION_STEM}.tsv"

log_line() {
  printf '%s\n' "$1"
  printf '%s\n' "$1" >> "$SUBMIT_LOG_FILE"
}

log_command() {
  local label="$1"
  shift
  printf '%s' "$label"
  printf '%s' "$label" >> "$SUBMIT_LOG_FILE"
  printf '%q ' "$@"
  printf '%q ' "$@" >> "$SUBMIT_LOG_FILE"
  printf '\n'
  printf '\n' >> "$SUBMIT_LOG_FILE"
}

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
DATASET_NAMES=(commonsense_qa commonsense_qa arc_challenge arc_challenge)
SOURCE_RUN_NAMES=(
  commonsense_qa_llama31_8b_allfamilies_probe_random_all_20260618
  commonsense_qa_qwen25_7b_allfamilies_probe_random_all_20260618
  arc_challenge_llama31_8b_allfamilies_probe_random_all_20260618
  arc_challenge_qwen25_7b_allfamilies_probe_random_all_20260618
)

TASK_FILTER="${TASK_FILTER:-}"
SELECTED_INDICES=()
task_filter_matches() {
  local label="${1:?missing task label}"
  local dataset_name="${2:?missing dataset name}"
  local compact_filter token
  if [[ -z "$TASK_FILTER" ]]; then
    return 0
  fi
  compact_filter="${TASK_FILTER//[[:space:]]/}"
  IFS=',' read -r -a filter_tokens <<< "$compact_filter"
  for token in "${filter_tokens[@]}"; do
    if [[ -n "$token" && ( "$token" == "$label" || "$token" == "$dataset_name" ) ]]; then
      return 0
    fi
  done
  return 1
}
for index in "${!TASK_LABELS[@]}"; do
  if task_filter_matches "${TASK_LABELS[$index]}" "${DATASET_NAMES[$index]}"; then
    SELECTED_INDICES+=("$index")
  fi
done
if [[ "${#SELECTED_INDICES[@]}" -eq 0 ]]; then
  printf '%s\n' "TASK_FILTER did not match a known task label or dataset: $TASK_FILTER" >&2
  exit 1
fi
SELECTED_BASE_INDICES_CSV="$(IFS=,; printf '%s' "${SELECTED_INDICES[*]}")"

require_any_file() {
  local label="${1:?missing file label}"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      return 0
    fi
  done
  printf '[preflight] missing %s; checked:' "$label" >&2
  printf ' %s' "$@" >&2
  printf '\n' >&2
  return 1
}

preflight_source_run() {
  local task_label="${1:?missing task label}"
  local run_dir="${2:?missing source run directory}"
  local chosen_probe_dir=""
  local candidate
  local failed=0
  if [[ ! -d "$run_dir" ]]; then
    printf '[preflight] missing source run task=%s path=%s\n' "$task_label" "$run_dir" >&2
    return 1
  fi
  require_any_file "run configuration for $task_label" \
    "$run_dir/meta/run_config.json" "$run_dir/run_config.json" || failed=1
  require_any_file "sampling records for $task_label" \
    "$run_dir/sampling/raw/sampling_records.jsonl" \
    "$run_dir/logs/sampling_records.jsonl" || failed=1
  require_any_file "probe prompt scores for $task_label" \
    "$run_dir/query/probe_scores_by_prompt.csv" \
    "$run_dir/probes/probe_scores_by_prompt.csv" || failed=1
  for candidate in \
    "$run_dir/probes/chosen/families/probe_bias_random_all" \
    "$run_dir/probes/chosen_probe/probe_bias_random_all"
  do
    if [[ -d "$candidate" ]]; then
      chosen_probe_dir="$candidate"
      break
    fi
  done
  if [[ -z "$chosen_probe_dir" ]]; then
    printf '[preflight] missing chosen random_all probe task=%s path=%s\n' \
      "$task_label" "$run_dir" >&2
    failed=1
  else
    require_any_file "chosen-probe metadata for $task_label" \
      "$chosen_probe_dir/metadata.json" || failed=1
    require_any_file "chosen-probe model for $task_label" \
      "$chosen_probe_dir/model.pkl" || failed=1
  fi
  if (( failed )); then
    return 1
  fi
  printf '[preflight] source_ok task=%s path=%s\n' "$task_label" "$run_dir"
}

printf 'array_task\ttask_label\tmodel_slug\tdataset\tsource_run_dir\tcell_root\n' > "$TASK_MATRIX"
SOURCE_PREFLIGHT_DIR="$SUBMIT_LOG_DIR/source_preflight_$SUBMISSION_STEM"
mkdir -p "$SOURCE_PREFLIGHT_DIR"
matrix_index=0
source_preflight_failed=0
for base_index in "${SELECTED_INDICES[@]}"; do
  source_run_dir="$SOURCE_RESULTS_ROOT/${MODEL_SLUGS[$base_index]}/${DATASET_NAMES[$base_index]}/${SOURCE_RUN_NAMES[$base_index]}"
  cell_root="$INTERVENTION_ROOT/${TASK_LABELS[$base_index]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$matrix_index" "${TASK_LABELS[$base_index]}" "${MODEL_SLUGS[$base_index]}" \
    "${DATASET_NAMES[$base_index]}" "$source_run_dir" "$cell_root" >> "$TASK_MATRIX"
  if ! preflight_source_run "${TASK_LABELS[$base_index]}" "$source_run_dir"; then
    source_preflight_failed=1
  elif ! "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" validate-source \
    --source-run-dir "$source_run_dir" \
    --output-dir "$SOURCE_PREFLIGHT_DIR/${TASK_LABELS[$base_index]}"
  then
    printf '[preflight] semantic source validation failed task=%s path=%s\n' \
      "${TASK_LABELS[$base_index]}" "$source_run_dir" >&2
    source_preflight_failed=1
  elif ! "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); required=("train","val","test"); counts=d.get("pairs_by_split",{}); missing=[s for s in required if int(counts.get(s,0)) <= 0]; sys.exit(f"missing positive paired split counts: {missing}") if missing else None' \
    "$SOURCE_PREFLIGHT_DIR/${TASK_LABELS[$base_index]}/source_validation.json"
  then
    printf '[preflight] incomplete paired splits task=%s path=%s\n' \
      "${TASK_LABELS[$base_index]}" "$source_run_dir" >&2
    source_preflight_failed=1
  fi
  matrix_index=$((matrix_index + 1))
done
if (( source_preflight_failed )); then
  printf '%s\n' \
    "Source preflight failed; no Slurm jobs were submitted. Fix SOURCE_RESULTS_ROOT or narrow TASK_FILTER." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
GPU_PARTITION="${GPU_PARTITION:-gpu,seas_gpu,gpu_h200}"
CPU_PARTITION="${CPU_PARTITION:-shared,sapphire}"
SBATCH_CPUS="${SBATCH_CPUS:-2}"
FIT_TIME="${FIT_TIME:-48:00:00}"
LOCALIZE_TIME="${LOCALIZE_TIME:-36:00:00}"
DOSE_TUNE_TIME="${DOSE_TUNE_TIME:-36:00:00}"
CONFIRM_TIME="${CONFIRM_TIME:-36:00:00}"
FIT_MEM="${FIT_MEM:-100G}"
LOCALIZE_MEM="${LOCALIZE_MEM:-100G}"
DOSE_TUNE_MEM="${DOSE_TUNE_MEM:-100G}"
CONFIRM_MEM="${CONFIRM_MEM:-100G}"
CPU_MEM="${CPU_MEM:-100G}"
selected_count="${#SELECTED_INDICES[@]}"
fit_array="0-$((selected_count - 1))"
localize_array="0-$((selected_count * 31 - 1))"
TOP_K_PATCH_LAYERS="${TOP_K_PATCH_LAYERS:-3}"
dose_tune_array="0-$((selected_count * TOP_K_PATCH_LAYERS - 1))"
cell_array="$fit_array"

export TASK_FILTER SELECTED_BASE_INDICES_CSV SOURCE_RESULTS_ROOT EXPERIMENT_RUN_ID ENV_PYTHON
export RUNTIME_ENV_DIR RUNTIME_CONTRACT_PATH
export INTERVENTION_BASE_ROOT INTERVENTION_ROOT TOP_K_PATCH_LAYERS SUBMISSION_GIT_COMMIT

fit_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_fit_${JOB_DATE_TAG}"
  --array "$fit_array"
  --partition "$GPU_PARTITION"
  --time "$FIT_TIME"
  --mem "$FIT_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/fit/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/fit/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/fit_directions_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] fit: " "${fit_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  fit_job_id="dryrun_fit_job_id"
else
  fit_job_id="$("${fit_cmd[@]}")"
fi

localize_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_loc_${JOB_DATE_TAG}"
  --dependency "afterok:$fit_job_id"
  --array "$localize_array"
  --partition "$GPU_PARTITION"
  --time "$LOCALIZE_TIME"
  --mem "$LOCALIZE_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/localize/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/localize/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/localize_layers_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] localize: " "${localize_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  localize_job_id="dryrun_localize_job_id"
else
  localize_job_id="$("${localize_cmd[@]}")"
fi

select_layers_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_lsel_${JOB_DATE_TAG}"
  --dependency "afterok:$localize_job_id"
  --array "$cell_array"
  --partition "$CPU_PARTITION"
  --time 04:00:00
  --mem 32G
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/select_layers/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/select_layers/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/select_layers_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] select-layers: " "${select_layers_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  select_layers_job_id="dryrun_select_layers_job_id"
else
  select_layers_job_id="$("${select_layers_cmd[@]}")"
fi

dose_tune_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_dose_${JOB_DATE_TAG}"
  --dependency "afterok:$select_layers_job_id"
  --array "$dose_tune_array"
  --partition "$GPU_PARTITION"
  --time "$DOSE_TUNE_TIME"
  --mem "$DOSE_TUNE_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/dose_tune/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/dose_tune/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/dose_tune_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] dose-tune: " "${dose_tune_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  dose_tune_job_id="dryrun_dose_tune_job_id"
else
  dose_tune_job_id="$("${dose_tune_cmd[@]}")"
fi

select_dose_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_dsel_${JOB_DATE_TAG}"
  --dependency "afterok:$dose_tune_job_id"
  --array "$cell_array"
  --partition "$CPU_PARTITION"
  --time 08:00:00
  --mem 64G
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/select_dose/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/select_dose/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/select_dose_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] select-dose: " "${select_dose_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  select_dose_job_id="dryrun_select_dose_job_id"
else
  select_dose_job_id="$("${select_dose_cmd[@]}")"
fi

confirm_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_test_${JOB_DATE_TAG}"
  --dependency "afterok:$select_dose_job_id"
  --array "$cell_array"
  --partition "$GPU_PARTITION"
  --time "$CONFIRM_TIME"
  --mem "$CONFIRM_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --output "$LOG_ROOT/slurm/confirm/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/confirm/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/confirm_test_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] confirm: " "${confirm_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  confirm_job_id="dryrun_confirm_job_id"
else
  confirm_job_id="$("${confirm_cmd[@]}")"
fi

aggregate_cmd=(
  sbatch --parsable
  --export=ALL
  --chdir "$REPO_DIR"
  --job-name "syco_int_agg_${JOB_DATE_TAG}"
  --dependency "afterok:$confirm_job_id"
  --array "$cell_array"
  --partition "$CPU_PARTITION"
  --time 08:00:00
  --mem "$CPU_MEM"
  --cpus-per-task 4
  --output "$LOG_ROOT/slurm/aggregate/%x.%A_%a.out"
  --error "$LOG_ROOT/slurm/aggregate/%x.%A_%a.err"
  "jobs/sycophancy_bias_probe/$BUNDLE_NAME/aggregate_array.sbatch"
)
log_command "[submit-$JOB_DATE_TAG] aggregate: " "${aggregate_cmd[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  aggregate_job_id="dryrun_aggregate_job_id"
else
  aggregate_job_id="$("${aggregate_cmd[@]}")"
fi

{
  printf 'BUNDLE_NAME=%q\n' "$BUNDLE_NAME"
  printf 'SUBMITTED_AT=%q\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'ROOT_DIR=%q\n' "$ROOT_DIR"
  printf 'REPO_DIR=%q\n' "$REPO_DIR"
  printf 'SUBMISSION_GIT_COMMIT=%q\n' "$SUBMISSION_GIT_COMMIT"
  printf 'ENV_PYTHON=%q\n' "$ENV_PYTHON"
  printf 'RUNTIME_ENV_DIR=%q\n' "$RUNTIME_ENV_DIR"
  printf 'RUNTIME_CONTRACT_PATH=%q\n' "$RUNTIME_CONTRACT_PATH"
  printf 'SOURCE_RESULTS_ROOT=%q\n' "$SOURCE_RESULTS_ROOT"
  printf 'EXPERIMENT_RUN_ID=%q\n' "$EXPERIMENT_RUN_ID"
  printf 'INTERVENTION_BASE_ROOT=%q\n' "$INTERVENTION_BASE_ROOT"
  printf 'INTERVENTION_ROOT=%q\n' "$INTERVENTION_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'TASK_FILTER=%q\n' "$TASK_FILTER"
  printf 'SELECTED_BASE_INDICES_CSV=%q\n' "$SELECTED_BASE_INDICES_CSV"
  printf 'TASK_MATRIX=%q\n' "$TASK_MATRIX"
  printf 'SOURCE_PREFLIGHT_DIR=%q\n' "$SOURCE_PREFLIGHT_DIR"
  printf 'TOP_K_PATCH_LAYERS=%q\n' "$TOP_K_PATCH_LAYERS"
  printf 'FIT_JOB_ID=%q\n' "$fit_job_id"
  printf 'LOCALIZE_JOB_ID=%q\n' "$localize_job_id"
  printf 'SELECT_LAYERS_JOB_ID=%q\n' "$select_layers_job_id"
  printf 'DOSE_TUNE_JOB_ID=%q\n' "$dose_tune_job_id"
  printf 'SELECT_DOSE_JOB_ID=%q\n' "$select_dose_job_id"
  printf 'CONFIRM_JOB_ID=%q\n' "$confirm_job_id"
  printf 'AGGREGATE_JOB_ID=%q\n' "$aggregate_job_id"
} > "$SUBMISSION_ENV_FILE"
cp "$SUBMISSION_ENV_FILE" "$LATEST_SUBMISSION_ENV_FILE"

log_line "[submit-$JOB_DATE_TAG] dry_run=$DRY_RUN selected_count=$selected_count"
log_line "[submit-$JOB_DATE_TAG] task_matrix=$TASK_MATRIX"
log_line "[submit-$JOB_DATE_TAG] intervention_root=$INTERVENTION_ROOT"
log_line "[submit-$JOB_DATE_TAG] fit_job_id=$fit_job_id"
log_line "[submit-$JOB_DATE_TAG] localize_job_id=$localize_job_id dependency=afterok:$fit_job_id"
log_line "[submit-$JOB_DATE_TAG] select_layers_job_id=$select_layers_job_id dependency=afterok:$localize_job_id"
log_line "[submit-$JOB_DATE_TAG] dose_tune_job_id=$dose_tune_job_id dependency=afterok:$select_layers_job_id"
log_line "[submit-$JOB_DATE_TAG] select_dose_job_id=$select_dose_job_id dependency=afterok:$dose_tune_job_id"
log_line "[submit-$JOB_DATE_TAG] confirm_job_id=$confirm_job_id dependency=afterok:$select_dose_job_id"
log_line "[submit-$JOB_DATE_TAG] aggregate_job_id=$aggregate_job_id dependency=afterok:$confirm_job_id"
log_line "[submit-$JOB_DATE_TAG] submission_env=$SUBMISSION_ENV_FILE"
