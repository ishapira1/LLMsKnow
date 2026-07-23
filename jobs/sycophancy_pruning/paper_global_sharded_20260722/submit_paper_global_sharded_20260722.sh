#!/bin/bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
STAGE="${1:-help}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export REPO_DIR="$ROOT_DIR"
cd "$ROOT_DIR"

BUNDLE_REL="jobs/sycophancy_pruning/paper_global_sharded_20260722"
BUNDLE_DIR="$ROOT_DIR/$BUNDLE_REL"
source "$BUNDLE_DIR/common.sh"

require_pushed_main_checkout() {
  local branch head upstream remote_main tracked_changes
  local protected_paths=(
    "$BUNDLE_REL"
    "tools/weight_pruning"
    "run_sycophancy_bias_probe.py"
    "scripts/aggregate_pruning_offline_evaluation.py"
    "scripts/build_pruning_result_package.py"
    "scripts/build_strict_sycophancy_manifests.py"
    "scripts/check_pruning_hard_stop.py"
    "scripts/collect_pruning_grid.py"
    "scripts/compare_pruning_masks.py"
    "scripts/prepare_weight_pruning_alpaca.py"
    "scripts/resolve_paper_pruning_artifact.py"
    "scripts/run_pruning_heldout_inference.py"
    "scripts/select_global_pruning_configuration.py"
    "scripts/snapshot_pruning_tokenization.py"
    "src/llmssycoph"
  )
  branch="$(git branch --show-current)"
  head="$(git rev-parse HEAD)"
  upstream="$(git rev-parse refs/remotes/origin/main)"
  remote_main="$(git ls-remote --exit-code origin refs/heads/main | awk 'NR == 1 {print $1}')"
  tracked_changes="$(git status --porcelain --untracked-files=all -- "${protected_paths[@]}")"
  if [[ "$branch" != "main" ]]; then
    printf '%s\n' "Harvard weight_pruning submissions require branch main, got: $branch" >&2
    exit 2
  fi
  if [[ -n "$tracked_changes" ]]; then
    printf '%s\n' "Harvard weight_pruning submissions require clean experiment source paths." >&2
    printf '%s\n' "$tracked_changes" >&2
    exit 2
  fi
  if [[ "$head" != "$upstream" ]]; then
    printf '%s\n' \
      "Harvard checkout is not synchronized: HEAD=$head origin/main=$upstream. Run git pull --ff-only origin main." >&2
    exit 2
  fi
  if [[ "$head" != "$remote_main" ]]; then
    printf '%s\n' \
      "Harvard checkout is not at the current remote main: HEAD=$head remote_main=${remote_main:-unknown}. Run git pull --ff-only origin main." >&2
    exit 2
  fi
}

ON_HARVARD=0
if command -v sbatch >/dev/null 2>&1 && command -v scontrol >/dev/null 2>&1; then
  require_harvard_scheduler
  require_pushed_main_checkout
  ON_HARVARD=1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  require_harvard_scheduler
fi

if [[ "$ON_HARVARD" == "1" ]]; then
  setup_environment
elif [[ "$DRY_RUN" == "1" ]]; then
  # A local dry run only renders commands. The mandatory storage, .env, and
  # scheduler preflight runs again on Harvard before any real submission.
  WEIGHT_PRUNING_LOG_ROOT="${TMPDIR:-/tmp}/LLMsKnow_weight_pruning_dry_run/$BUNDLE_NAME"
  EXPERIMENT_ROOT="${TMPDIR:-/tmp}/LLMsKnow_weight_pruning_dry_run/$BUNDLE_NAME/experiment"
  export WEIGHT_PRUNING_LOG_ROOT EXPERIMENT_ROOT
  printf '%s\n' \
    "[weight_pruning] local DRY_RUN: command rendering only; Harvard preflight is not simulated." >&2
else
  printf '%s\n' "weight_pruning jobs may only be submitted through Harvard Slurm." >&2
  exit 2
fi

LOG_ROOT="$WEIGHT_PRUNING_LOG_ROOT"
mkdir -p \
  "$LOG_ROOT/submit" \
  "$LOG_ROOT/slurm/prepare" \
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
SBATCH_SUBMIT_DELAY_SECONDS="${SBATCH_SUBMIT_DELAY_SECONDS:-1}"
SBATCH_BASE=(sbatch --parsable --chdir="$ROOT_DIR" --export=ALL)

log_line() {
  printf '%s\n' "$1"
  printf '%s\n' "$1" >> "$SUBMIT_LOG"
}

log_detail() {
  printf '%s\n' "$1" >&2
  printf '%s\n' "$1" >> "$SUBMIT_LOG"
}

safe_job_component() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  value="${value//+/_}"
  value="${value//\//_}"
  value="${value//,/_}"
  printf '%s\n' "$value"
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
    log_detail "[submit] rendered description=$description dry_id=dry_$dry_id"
    printf 'dry_%s\n' "$dry_id"
    return
  fi

  local raw_id job_id
  raw_id="$("$@")"
  job_id="${raw_id%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "Unexpected sbatch --parsable response for $description: $raw_id" >&2
    exit 1
  fi
  log_detail "[submit] confirmed description=$description job_id=$job_id"
  printf '%s\n' "$job_id"
  sleep "$SBATCH_SUBMIT_DELAY_SECONDS"
}

submit_prepare() {
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name=weight_pruning_prepare
    --output="$LOG_ROOT/slurm/prepare/%x.%j.out"
    --error="$LOG_ROOT/slurm/prepare/%x.%j.err"
    "$BUNDLE_DIR/prepare_data.sbatch"
  )
  submit_job prepare_data "${cmd[@]}"
}

submit_sampling() {
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name=weight_pruning_sample
    --output="$LOG_ROOT/slurm/sampling/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/sampling/%x.%A_%a.err"
    --array=0-11
    "$BUNDLE_DIR/sampling_array.sbatch"
  )
  submit_job sampling "${cmd[@]}"
}

submit_manifests() {
  local dependency="${1:-}"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name=weight_pruning_manifest
    --output="$LOG_ROOT/slurm/manifests/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/manifests/%x.%A_%a.err"
    --array=0-1
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/manifests_array.sbatch")
  submit_job manifests "${cmd[@]}"
}

submit_tokens() {
  local dependency="${1:-}"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name=weight_pruning_tokens
    --output="$LOG_ROOT/slurm/token_snapshots/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/token_snapshots/%x.%A_%a.err"
    --array=0-1
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/token_snapshot_array.sbatch")
  submit_job token_snapshots "${cmd[@]}"
}

submit_scores() {
  local size="$1"
  local variant="$2"
  local seeds_csv="$3"
  local dependency="${4:-}"
  local variant_slug
  variant_slug="$(safe_job_component "$variant")"
  IFS=, read -r -a seeds <<< "$seeds_csv"
  local total=$((2 * ${#seeds[@]} * 2))
  export MANIFEST_SIZE="$size"
  export SCORE_VARIANT="$variant"
  export CALIBRATION_SEEDS_CSV="$seeds_csv"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_${variant_slug}_score"
    --output="$LOG_ROOT/slurm/scores/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/scores/%x.%A_%a.err"
    --array="0-$((total - 1))"
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/score_array.sbatch")
  log_detail "[submit] score_env size=$size variant=$variant seeds=$seeds_csv"
  submit_job "scores_${size}_${variant}_${seeds_csv}" "${cmd[@]}"
}

submit_grid_baseline() {
  local size="$1"
  local splits="$2"
  local dependency="${3:-}"
  local splits_slug
  splits_slug="$(safe_job_component "$splits")"
  export MANIFEST_SIZE="$size"
  export GRID_Q=0
  export GRID_EVAL_SPLITS="$splits"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_${splits_slug}_base"
    --output="$LOG_ROOT/slurm/grid/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/grid/%x.%A_%a.err"
    --array=0-1
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_eval_grid_array.sbatch")
  log_detail "[submit] grid_baseline_env size=$size splits=$splits"
  submit_job "grid_baseline_${size}_${splits}" "${cmd[@]}"
}

submit_grid_q() {
  local size="$1"
  local q="$2"
  local dependency="${3:-}"
  local q_slug
  q_slug="$(safe_job_component "$q")"
  export MANIFEST_SIZE="$size"
  export GRID_Q="$q"
  export GRID_EVAL_SPLITS=val
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_q_${q_slug}"
    --output="$LOG_ROOT/slurm/grid/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/grid/%x.%A_%a.err"
    --array=0-9
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_eval_grid_array.sbatch")
  log_detail "[submit] grid_q_env size=$size q=$q"
  submit_job "grid_${size}_q_${q}" "${cmd[@]}"
}

submit_selection() {
  local size="$1"
  local dependency="${2:-}"
  export MANIFEST_SIZE="$size"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_select"
    --output="$LOG_ROOT/slurm/selection/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/selection/%x.%A_%a.err"
    --array=0-1
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/select_array.sbatch")
  submit_job "selection_${size}" "${cmd[@]}"
}

submit_selected() {
  local size="$1"
  local variants="$2"
  local splits="$3"
  local dependency="${4:-}"
  local stage_label="${5:-selected}"
  local stage_slug splits_slug
  stage_slug="$(safe_job_component "$stage_label")"
  splits_slug="$(safe_job_component "$splits")"
  IFS=, read -r -a variant_array <<< "$variants"
  local total=$((2 * ${#variant_array[@]}))
  export MANIFEST_SIZE="$size"
  export SELECTED_VARIANTS_CSV="$variants"
  export EVAL_SPLITS="$splits"
  local array_spec="0-$((total - 1))"
  if [[ -n "${SELECTED_ARRAY_CONCURRENCY:-}" ]]; then
    array_spec+="%${SELECTED_ARRAY_CONCURRENCY}"
  fi
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_${stage_slug}_${splits_slug}"
    --output="$LOG_ROOT/slurm/selected/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/selected/%x.%A_%a.err"
    --array="$array_spec"
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/selected_mask_eval_array.sbatch")
  log_detail "[submit] selected_env size=$size variants=$variants splits=$splits stage=$stage_label"
  submit_job "selected_${stage_label}_${splits}" "${cmd[@]}"
}

submit_mask_overlap() {
  local size="$1"
  local dependency="${2:-}"
  export MANIFEST_SIZE="$size"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name="weight_pruning_${size}_overlap"
    --output="$LOG_ROOT/slurm/overlap/%x.%A_%a.out"
    --error="$LOG_ROOT/slurm/overlap/%x.%A_%a.err"
    --array=0-1
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/mask_overlap_array.sbatch")
  submit_job "mask_overlap_${size}" "${cmd[@]}"
}

submit_report() {
  local dependency="${1:-}"
  local cmd=(
    "${SBATCH_BASE[@]}"
    --job-name=weight_pruning_report
    --output="$LOG_ROOT/slurm/report/%x.%j.out"
    --error="$LOG_ROOT/slurm/report/%x.%j.err"
  )
  if [[ -n "$dependency" ]]; then cmd+=("--dependency=afterok:$dependency"); fi
  cmd+=("$BUNDLE_DIR/report.sbatch")
  log_detail "[submit] report_env output_dir=${REPORT_OUTPUT_DIR:-<experiment-root>/reports/minimum_result_package} bootstrap_seed=${REPORT_BOOTSTRAP_SEED:-5} overwrite=${REPORT_OVERWRITE:-0}"
  submit_job report "${cmd[@]}"
}

write_setup_record() {
  local prepare_id="$1"
  local sampling_id="$2"
  local manifest_id="$3"
  local token_id="$4"
  if [[ "$DRY_RUN" == "1" ]]; then
    log_line "[submit] dry_run_setup_record=not_written"
    return
  fi
  local record="$LOG_ROOT/submit/setup_$(date +%Y%m%dT%H%M%S%z)_jobs.env"
  local git_commit
  git_commit="$(git rev-parse HEAD)"
  {
    printf 'RUN_KIND=%q\n' "weight_pruning_setup"
    printf 'DRY_RUN=%q\n' "0"
    printf 'GIT_COMMIT=%q\n' "$git_commit"
    printf 'PREPARE_JOB_ID=%q\n' "$prepare_id"
    printf 'SAMPLING_JOB_ID=%q\n' "$sampling_id"
    printf 'MANIFEST_JOB_ID=%q\n' "$manifest_id"
    printf 'TOKEN_JOB_ID=%q\n' "$token_id"
    printf 'EXPERIMENT_ROOT=%q\n' "$EXPERIMENT_ROOT"
    printf 'WEIGHT_PRUNING_LOG_ROOT=%q\n' "$LOG_ROOT"
  } > "$record"
  cp "$record" "$LOG_ROOT/submit/latest_setup.env"
  log_line "[submit] setup_record=$record latest=$LOG_ROOT/submit/latest_setup.env"
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

log_line "[submit] stage=$STAGE dry_run=$DRY_RUN repo=$ROOT_DIR log=$SUBMIT_LOG"

case "$STAGE" in
  setup)
    prepare_id="$(submit_prepare)"
    sampling_id="$(submit_sampling)"
    setup_dependency="$prepare_id:$sampling_id"
    manifest_id="$(submit_manifests "$setup_dependency")"
    token_id="$(submit_tokens "$manifest_id")"
    log_line "[submit] setup prepare=$prepare_id sampling=$sampling_id manifests=$manifest_id tokens=$token_id"
    write_setup_record "$prepare_id" "$sampling_id" "$manifest_id" "$token_id"
    ;;
  prepare-data)
    submit_prepare
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
      val "$dependency" controls
    ;;
  replications)
    score_id="$(submit_scores main primary 17,29 "${DEPENDENCY_JOB_ID:-}")"
    replication_id="$(submit_selected main primary_seed17,primary_seed29 val "$score_id" replications)"
    submit_mask_overlap main "$replication_id"
    ;;
  final-test)
    export RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-1}"
    export RUN_ALPACA_EVAL="${RUN_ALPACA_EVAL:-1}"
    export SAVE_SELECTED_MODEL="${SAVE_SELECTED_MODEL:-1}"
    export SELECTED_ARRAY_CONCURRENCY="${SELECTED_ARRAY_CONCURRENCY:-1}"
    baseline_id="$(submit_grid_baseline main test "${DEPENDENCY_JOB_ID:-}")"
    final_id="$(submit_selected main primary,primary_seed17,primary_seed29 test "$baseline_id" final)"
    report_id="$(submit_report "$final_id")"
    log_line "[submit] final-test baseline=$baseline_id selected=$final_id report=$report_id"
    ;;
  selected)
    submit_selected \
      "${MANIFEST_SIZE:-main}" \
      "${SELECTED_VARIANTS_CSV:-primary}" \
      "${EVAL_SPLITS:-val}" \
      "${DEPENDENCY_JOB_ID:-}" \
      selected
    ;;
  report)
    submit_report "${DEPENDENCY_JOB_ID:-}"
    ;;
  help|*)
    printf '%s\n' \
      "Usage: $0 {setup|prepare-data|sampling|manifests|tokens|score|grid-baseline|grid-q|smoke|pilot|main|select|controls|replications|final-test|selected|report}" \
      "" \
      "Run from the single LLMsKnow checkout. Model revisions are pinned in common.sh." \
      "Use DRY_RUN=1 to render commands; on Harvard it also validates .env, storage, and the odyssey scheduler." \
      "Use DEPENDENCY_JOB_ID=<id> to chain a standalone stage."
    if [[ "$STAGE" != "help" ]]; then exit 2; fi
    ;;
esac
