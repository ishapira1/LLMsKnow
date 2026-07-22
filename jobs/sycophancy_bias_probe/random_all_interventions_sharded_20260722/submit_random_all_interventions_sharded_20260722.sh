#!/bin/bash

set -euo pipefail

BUNDLE_NAME="random_all_interventions_sharded_20260722"
JOB_DATE_TAG="20260722"
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
for index in "${!TASK_LABELS[@]}"; do
  if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "${TASK_LABELS[$index]}" ]]; then
    SELECTED_INDICES+=("$index")
  fi
done
if [[ "${#SELECTED_INDICES[@]}" -eq 0 ]]; then
  printf '%s\n' "TASK_FILTER did not match a known task: $TASK_FILTER" >&2
  exit 1
fi

printf 'array_task\ttask_label\tmodel_slug\tdataset\tsource_run_dir\tcell_root\n' > "$TASK_MATRIX"
matrix_index=0
for base_index in "${SELECTED_INDICES[@]}"; do
  source_run_dir="$SOURCE_RESULTS_ROOT/${MODEL_SLUGS[$base_index]}/${DATASET_NAMES[$base_index]}/${SOURCE_RUN_NAMES[$base_index]}"
  cell_root="$INTERVENTION_ROOT/${TASK_LABELS[$base_index]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$matrix_index" "${TASK_LABELS[$base_index]}" "${MODEL_SLUGS[$base_index]}" \
    "${DATASET_NAMES[$base_index]}" "$source_run_dir" "$cell_root" >> "$TASK_MATRIX"
  matrix_index=$((matrix_index + 1))
done

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

export TASK_FILTER SOURCE_RESULTS_ROOT EXPERIMENT_RUN_ID INTERVENTION_BASE_ROOT INTERVENTION_ROOT TOP_K_PATCH_LAYERS

fit_cmd=(
  sbatch --parsable
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
  printf 'SOURCE_RESULTS_ROOT=%q\n' "$SOURCE_RESULTS_ROOT"
  printf 'EXPERIMENT_RUN_ID=%q\n' "$EXPERIMENT_RUN_ID"
  printf 'INTERVENTION_BASE_ROOT=%q\n' "$INTERVENTION_BASE_ROOT"
  printf 'INTERVENTION_ROOT=%q\n' "$INTERVENTION_ROOT"
  printf 'LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'TASK_FILTER=%q\n' "$TASK_FILTER"
  printf 'TASK_MATRIX=%q\n' "$TASK_MATRIX"
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
