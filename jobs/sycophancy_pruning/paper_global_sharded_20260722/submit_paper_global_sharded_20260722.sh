#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BUNDLE_DIR="jobs/sycophancy_pruning/paper_global_sharded_20260722"
LOG_ROOT="jobs/sycophancy_pruning/logs/paper_global_sharded_20260722"
mkdir -p \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/sampling" \
  "$LOG_ROOT/slurm/manifests" \
  "$LOG_ROOT/slurm/token_snapshots" \
  "$LOG_ROOT/slurm/scores" \
  "$LOG_ROOT/slurm/grid" \
  "$LOG_ROOT/slurm/selection" \
  "$LOG_ROOT/slurm/selected" \
  "$LOG_ROOT/slurm/overlap" \
  "$LOG_ROOT/slurm/report" \
  "$LOG_ROOT/by_task"

SUBMIT_LOG="$LOG_ROOT/submit/submit_$(date +%Y%m%dT%H%M%S%z)_pid_$$.log"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${1:-help}"
shift || true

log_line() {
  printf '%s\n' "$1"
  printf '%s\n' "$1" >> "$SUBMIT_LOG"
}

log_detail() {
  printf '%s\n' "$1" >&2
  printf '%s\n' "$1" >> "$SUBMIT_LOG"
}

submit_job() {
  local description="$1"
  shift
  printf '[submit] %s: ' "$description" >&2
  printf '%q ' "$@" >&2
  printf '\n' >&2
  {
    printf '[submit] %s: ' "$description"
    printf '%q ' "$@"
    printf '\n'
  } >> "$SUBMIT_LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    local dry_id="${description//[^[:alnum:]_]/_}"
    printf 'dry_%s\n' "$dry_id"
  else
    "$@"
  fi
}

dependency_args() {
  local job_id="${1:-}"
  if [[ -n "$job_id" ]]; then
    printf '%s\n' "--dependency=afterok:$job_id"
  fi
}

submit_sampling() {
  submit_job sampling sbatch --parsable --array 0-11 "$BUNDLE_DIR/sampling_array.sbatch"
}

submit_manifests() {
  local dependency="${1:-}"
  local cmd=(sbatch --parsable --array 0-1)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/manifests_array.sbatch")
  submit_job manifests "${cmd[@]}"
}

submit_tokens() {
  local dependency="${1:-}"
  local cmd=(sbatch --parsable --array 0-1)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/token_snapshot_array.sbatch")
  submit_job token_snapshots "${cmd[@]}"
}

submit_scores() {
  local size="$1"
  local variant="$2"
  local seeds_csv="$3"
  local dependency="${4:-}"
  IFS=, read -r -a seeds <<< "$seeds_csv"
  local total=$((2 * ${#seeds[@]} * 2))
  export MANIFEST_SIZE="$size"
  export SCORE_VARIANT="$variant"
  export CALIBRATION_SEEDS_CSV="$seeds_csv"
  local cmd=(sbatch --parsable --array "0-$((total - 1))")
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/score_array.sbatch")
  log_detail "[submit] score_env size=$size variant=$variant seeds=$seeds_csv"
  submit_job "scores_${size}_${variant}_${seeds_csv}" "${cmd[@]}"
}

submit_grid_baseline() {
  local size="$1"
  local splits="$2"
  local dependency="${3:-}"
  export MANIFEST_SIZE="$size"
  export GRID_Q=0
  export GRID_EVAL_SPLITS="$splits"
  local cmd=(sbatch --parsable --array 0-1)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_eval_grid_array.sbatch")
  log_detail "[submit] grid_baseline_env size=$size splits=$splits"
  submit_job "grid_baseline_${size}_${splits}" "${cmd[@]}"
}

submit_grid_q() {
  local size="$1"
  local q="$2"
  local dependency="${3:-}"
  export MANIFEST_SIZE="$size"
  export GRID_Q="$q"
  export GRID_EVAL_SPLITS=val
  local cmd=(sbatch --parsable --array 0-9)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_eval_grid_array.sbatch")
  log_detail "[submit] grid_q_env size=$size q=$q"
  submit_job "grid_${size}_q_${q}" "${cmd[@]}"
}

submit_selection() {
  local size="$1"
  local dependency="${2:-}"
  export MANIFEST_SIZE="$size"
  local cmd=(sbatch --parsable --array 0-1)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/select_array.sbatch")
  submit_job "selection_${size}" "${cmd[@]}"
}

submit_selected() {
  local size="$1"
  local variants="$2"
  local splits="$3"
  local dependency="${4:-}"
  IFS=, read -r -a variant_array <<< "$variants"
  local total=$((2 * ${#variant_array[@]}))
  export MANIFEST_SIZE="$size"
  export SELECTED_VARIANTS_CSV="$variants"
  export EVAL_SPLITS="$splits"
  local array_spec="0-$((total - 1))"
  if [[ -n "${SELECTED_ARRAY_CONCURRENCY:-}" ]]; then
    array_spec+="%${SELECTED_ARRAY_CONCURRENCY}"
  fi
  local cmd=(sbatch --parsable --array "$array_spec")
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/selected_mask_eval_array.sbatch")
  log_detail "[submit] selected_env size=$size variants=$variants splits=$splits"
  submit_job "selected_${splits}" "${cmd[@]}"
}

submit_mask_overlap() {
  local size="$1"
  local dependency="${2:-}"
  export MANIFEST_SIZE="$size"
  local cmd=(sbatch --parsable --array 0-1)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_overlap_array.sbatch")
  submit_job "mask_overlap_${size}" "${cmd[@]}"
}

submit_report() {
  local dependency="${1:-}"
  local cmd=(sbatch --parsable)
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/report.sbatch")
  log_detail "[submit] report_env output_dir=${REPORT_OUTPUT_DIR:-<experiment-root>/reports/minimum_result_package} bootstrap_seed=${REPORT_BOOTSTRAP_SEED:-5} overwrite=${REPORT_OVERWRITE:-0}"
  submit_job report "${cmd[@]}"
}

submit_tier_chain() {
  local size="$1"
  local q_values
  case "$size" in
    smoke) q_values="${Q_WAVES_CSV:-1e-6,1e-5,5e-5}" ;;
    pilot) q_values="${Q_WAVES_CSV:-1e-6,3e-6,7e-6,1e-5,2e-5,5e-5}" ;;
    main) q_values="${Q_WAVES_CSV:-1e-6,3e-6,7e-6,1e-5,2e-5,5e-5,1e-4}" ;;
    *) printf '%s\n' "Unknown tier: $size" >&2; exit 2 ;;
  esac
  local score_id baseline_id previous_id selection_id
  score_id="$(submit_scores "$size" primary 5 "${DEPENDENCY_JOB_ID:-}")"
  baseline_id="$(submit_grid_baseline "$size" val "$score_id")"
  previous_id="$baseline_id"
  IFS=, read -r -a q_array <<< "$q_values"
  for q in "${q_array[@]}"; do
    previous_id="$(submit_grid_q "$size" "$q" "$previous_id")"
  done
  selection_id="$(submit_selection "$size" "$previous_id")"
  log_line "[submit] tier=$size score=$score_id baseline=$baseline_id final_q=$previous_id selection=$selection_id"
}

log_line "[submit] stage=$STAGE dry_run=$DRY_RUN log=$SUBMIT_LOG"

case "$STAGE" in
  setup)
    sampling_id="$(submit_sampling)"
    manifest_id="$(submit_manifests "$sampling_id")"
    token_id="$(submit_tokens "$manifest_id")"
    log_line "[submit] setup sampling=$sampling_id manifests=$manifest_id tokens=$token_id"
    ;;
  sampling)
    submit_sampling
    ;;
  manifests)
    submit_manifests "${DEPENDENCY_JOB_ID:-}"
    ;;
  tokens)
    submit_tokens "${DEPENDENCY_JOB_ID:-}"
    ;;
  score)
    submit_scores "${MANIFEST_SIZE:-main}" "${SCORE_VARIANT:-primary}" "${CALIBRATION_SEEDS_CSV:-5}" "${DEPENDENCY_JOB_ID:-}"
    ;;
  grid-baseline)
    submit_grid_baseline "${MANIFEST_SIZE:-main}" "${GRID_EVAL_SPLITS:-val}" "${DEPENDENCY_JOB_ID:-}"
    ;;
  grid-q)
    if [[ -z "${GRID_Q:-}" ]]; then
      printf '%s\n' "grid-q requires GRID_Q" >&2
      exit 2
    fi
    submit_grid_q "${MANIFEST_SIZE:-main}" "$GRID_Q" "${DEPENDENCY_JOB_ID:-}"
    ;;
  select)
    submit_selection "${MANIFEST_SIZE:-main}" "${DEPENDENCY_JOB_ID:-}"
    ;;
  smoke|pilot|main)
    submit_tier_chain "$STAGE"
    ;;
  controls)
    score_ids=()
    for variant in structure_matched alpaca_only chat choice_token released_abs; do
      score_ids+=("$(submit_scores main "$variant" 5 "${DEPENDENCY_JOB_ID:-}")")
    done
    dependency="$(IFS=:; printf '%s' "${score_ids[*]}")"
    submit_selected main \
      primary,structure_matched,opposite_sign,second_slice,random_magnitude,alpaca_only,chat,choice_token,released_abs \
      val "$dependency"
    ;;
  replications)
    score_id="$(submit_scores main primary 17,29 "${DEPENDENCY_JOB_ID:-}")"
    replication_id="$(submit_selected main primary_seed17,primary_seed29 val "$score_id")"
    submit_mask_overlap main "$replication_id"
    ;;
  final-test)
    export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-1}"
    export RUN_ALPACA_EVAL="${RUN_ALPACA_EVAL:-1}"
    export SAVE_SELECTED_MODEL="${SAVE_SELECTED_MODEL:-1}"
    export SELECTED_ARRAY_CONCURRENCY="${SELECTED_ARRAY_CONCURRENCY:-1}"
    baseline_id="$(submit_grid_baseline main test "${DEPENDENCY_JOB_ID:-}")"
    final_id="$(submit_selected main primary,primary_seed17,primary_seed29 test "$baseline_id")"
    report_id="$(submit_report "$final_id")"
    log_line "[submit] final-test baseline=$baseline_id selected=$final_id report=$report_id"
    ;;
  selected)
    submit_selected "${MANIFEST_SIZE:-main}" "${SELECTED_VARIANTS_CSV:-primary}" "${EVAL_SPLITS:-val}" "${DEPENDENCY_JOB_ID:-}"
    ;;
  report)
    submit_report "${DEPENDENCY_JOB_ID:-}"
    ;;
  help|*)
    printf '%s\n' \
      "Usage: $0 {setup|sampling|manifests|tokens|score|grid-baseline|grid-q|smoke|pilot|main|select|controls|replications|final-test|selected|report}" \
      "" \
      "Set QWEN_REVISION and LLAMA_REVISION to immutable commit SHAs." \
      "Use DRY_RUN=1 for validation and DEPENDENCY_JOB_ID=<id> to chain a standalone stage."
    if [[ "$STAGE" != "help" ]]; then exit 2; fi
    ;;
esac
