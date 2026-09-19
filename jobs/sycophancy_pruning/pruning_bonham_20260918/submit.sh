#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "$0")" && pwd -P)/common.sh"

DRY_RUN="${DRY_RUN:-1}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { printf 'DRY_RUN must be 0 or 1\n' >&2; exit 2; }
if [[ "$DRY_RUN" == 0 ]]; then require_runtime; fi
mkdir -p "$RESULT_ROOT" "$LOG_ROOT/submit" "$LOG_ROOT/slurm" "$LOG_ROOT/by_task"
submission_log="$LOG_ROOT/submit/submit_$(date +%Y%m%dT%H%M%S).log"

submit_job() {
  label="$1"; stage="$2"; script="$3"; dependency="$4"; array="$5"; model_key="$6"; partition="$7"; gres="$8"
  mkdir -p "$LOG_ROOT/slurm/$stage"
  command=(sbatch --parsable --kill-on-invalid-dep=yes --job-name "$label" --export "ALL,STAGE=$stage,MODEL_KEY=$model_key,BONHAM_BUNDLE_DIR=$BUNDLE_DIR" --output "$LOG_ROOT/slurm/$stage/%x_%A_%a.out" --error "$LOG_ROOT/slurm/$stage/%x_%A_%a.err")
  [[ -z "$dependency" ]] || command+=(--dependency "afterok:$dependency")
  [[ -z "$array" ]] || command+=(--array "$array")
  [[ -z "$partition" ]] || command+=(--partition "$partition")
  [[ -z "$gres" ]] || command+=(--gres "$gres")
  case "$stage" in
    model_smoke) command+=(--time 00:10:00 --mem 48G) ;;
    score_component) command+=(--time 24:00:00 --mem 120G) ;;
    neutral_screen) command+=(--time 00:10:00 --mem 48G) ;;
    n1_screen|source_screen) command+=(--time 06:00:00) ;;
    eval_generalization|eval_useful_assertions|eval_source_attribution|eval_capabilities) command+=(--time 05:00:00) ;;
    evalplus_run) command+=(--time 03:00:00 --cpus-per-task 8 --mem 24G) ;;
  esac
  command+=("$script")
  print_command "${command[@]}" | tee -a "$submission_log" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'dry_%s\n' "$label"
  else
    raw_job_id="$("${command[@]}")"
    job_id="${raw_job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { printf 'unexpected sbatch response: %s\n' "$raw_job_id" >&2; return 2; }
    printf 'submitted label=%s stage=%s job_id=%s\n' "$label" "$stage" "$job_id" | tee -a "$submission_log" >&2
    printf '%s\n' "$job_id"
  fi
}

reuse_root_job() {
  label="$1"; job_id="$2"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf 'invalid reusable %s job id: %s\n' "$label" "$job_id" >&2
    return 2
  }
  state=unchecked_dry_run
  if [[ "$DRY_RUN" == 0 ]]; then
    state="$(sacct -X -n -j "$job_id" --format=State | awk 'NF { sub(/\+.*/, "", $1); print $1; exit }')"
    case "$state" in
      COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED) ;;
      '') printf 'reusable %s job %s is unknown to Slurm accounting\n' "$label" "$job_id" >&2; return 2 ;;
      *) printf 'reusable %s job %s has unusable state %s\n' "$label" "$job_id" "$state" >&2; return 2 ;;
    esac
  fi
  printf 'reused label=%s job_id=%s state=%s\n' "$label" "$job_id" "$state" | tee -a "$submission_log" >&2
  printf '%s\n' "$job_id"
}

cpu="$BUNDLE_DIR/cpu_stage.sbatch"
gpu="$BUNDLE_DIR/gpu_array.sbatch"

if [[ -n "${BONHAM_REUSE_CAPABILITY_SOURCES_JOB_ID:-}" ]]; then
  cap_sources="$(reuse_root_job capability_sources "$BONHAM_REUSE_CAPABILITY_SOURCES_JOB_ID")"
else
  cap_sources="$(submit_job bonh_capsrc_0918 capability_sources "$cpu" '' '' shared '' '')"
fi
if [[ -n "${BONHAM_REUSE_SOURCE_FREEZE_JOB_ID:-}" ]]; then
  source_freeze="$(reuse_root_job source_freeze "$BONHAM_REUSE_SOURCE_FREEZE_JOB_ID")"
else
  source_freeze="$(submit_job bonh_freeze_0918 source_freeze "$cpu" '' '' shared '' '')"
fi

models=(llama31_8b qwen25_7b gemma4_12b)
smokes=(); neutrals=()
for index in 0 1 2; do
  model="${models[$index]}"
  if [[ "$model" == qwen25_7b ]]; then
    partition=gpu,gpu_requeue
    gres=gpu:nvidia_a100-sxm4-80gb:1
  else
    partition=gpu_h200,gpu_requeue
    gres=gpu:nvidia_h200:1
  fi
  smoke_partition="$partition"
  smokes[$index]="$(submit_job "bonh_${model}_smk" model_smoke "$gpu" "$source_freeze" '' "$model" "$smoke_partition" "$gres")"
  neutrals[$index]="$(submit_job "bonh_${model}_neu" neutral_screen "$gpu" "${smokes[$index]}" '0-79%16' "$model" "$partition" "$gres")"
done

prepare_screens="$(submit_job bonh_screenprep prepare_screens "$cpu" "${neutrals[0]}:${neutrals[1]}:${neutrals[2]}" '' shared '' '')"

n1_screens=(); source_screens=()
for index in 0 1 2; do
  model="${models[$index]}"
  if [[ "$model" == qwen25_7b ]]; then
    partition=gpu,gpu_requeue
    gres=gpu:nvidia_a100-sxm4-80gb:1
  else
    partition=gpu_h200,gpu_requeue
    gres=gpu:nvidia_h200:1
  fi
  n1_screens[$index]="$(submit_job "bonh_${model}_n1scr" n1_screen "$gpu" "$prepare_screens" '0-319%16' "$model" "$partition" "$gres")"
  source_screens[$index]="$(submit_job "bonh_${model}_srcscr" source_screen "$gpu" "$prepare_screens" '0-47%16' "$model" "$partition" "$gres")"
done

screen_dependencies="${n1_screens[0]}:${n1_screens[1]}:${n1_screens[2]}:${source_screens[0]}:${source_screens[1]}:${source_screens[2]}"
allocate="$(submit_job bonh_allocate_0918 allocate_manifests "$cpu" "$screen_dependencies" '' shared '' '')"
eval_prepare="$(submit_job bonh_evalprep_0918 eval_prepare "$cpu" "$allocate:$cap_sources" '' shared '' '')"
source_attribution_prepare="$(submit_job bonh_srcprep_0918 source_attribution_prepare "$cpu" "$eval_prepare" '' shared '' '')"

scores=(); masks=(); randoms=(); states=(); steer_preps=(); steer_extracts=(); steer_develops=(); weights=()
for index in 0 1 2; do
  model="${models[$index]}"
  if [[ "$model" == qwen25_7b ]]; then
    partition=gpu,gpu_requeue
    gres=gpu:nvidia_a100-sxm4-80gb:1
  else
    partition=gpu_h200,gpu_requeue
    gres=gpu:nvidia_h200:1
  fi
  scores[$index]="$(submit_job "bonh_${model}_score" score_component "$gpu" "$allocate" '0-6%7' "$model" "$partition" "$gres")"
  masks[$index]="$(submit_job "bonh_${model}_mask" build_masks "$cpu" "${scores[$index]}" '' "$model" '' '')"
  randoms[$index]="$(submit_job "bonh_${model}_rand" random_masks "$gpu" "${masks[$index]}" '' "$model" "$partition" "$gres")"
  states[$index]="$(submit_job "bonh_${model}_state" build_states "$cpu" "${randoms[$index]}" '' "$model" '' '')"
  steer_preps[$index]="$(submit_job "bonh_${model}_stprep" steering_prepare "$cpu" "$allocate" '' "$model" '' '')"
  steer_extracts[$index]="$(submit_job "bonh_${model}_stext" steering_extract "$gpu" "${steer_preps[$index]}" '' "$model" "$partition" "$gres")"
  steer_develops[$index]="$(submit_job "bonh_${model}_stdev" steering_develop "$gpu" "${steer_extracts[$index]}" '' "$model" "$partition" "$gres")"
  weights[$index]="$(submit_job "bonh_${model}_weight" weight_model "$cpu" "${masks[$index]}" '' "$model" '' '')"
done

weight_aggregate="$(submit_job bonh_weightagg_0918 weight_aggregate "$cpu" "${weights[0]}:${weights[1]}:${weights[2]}" '' shared '' '')"

eval_general=(); eval_useful=(); eval_source=(); eval_caps=()
for index in 0 1 2; do
  model="${models[$index]}"
  if [[ "$model" == qwen25_7b ]]; then
    partition=gpu,gpu_requeue
    gres=gpu:nvidia_a100-sxm4-80gb:1
  else
    partition=gpu_h200,gpu_requeue
    gres=gpu:nvidia_h200:1
  fi
  eval_dependency="$eval_prepare:${states[$index]}:${steer_develops[$index]}"
  eval_general[$index]="$(submit_job "bonh_${model}_gen" eval_generalization "$gpu" "$eval_dependency" '0-479%16' "$model" "$partition" "$gres")"
  eval_useful[$index]="$(submit_job "bonh_${model}_use" eval_useful_assertions "$gpu" "$eval_dependency" '0-959%16' "$model" "$partition" "$gres")"
  eval_source[$index]="$(submit_job "bonh_${model}_srcattr" eval_source_attribution "$gpu" "$source_attribution_prepare:${states[$index]}:${steer_develops[$index]}" '0-479%16' "$model" "$partition" "$gres")"
  eval_caps[$index]="$(submit_job "bonh_${model}_cap" eval_capabilities "$gpu" "$eval_dependency" '0-639%16' "$model" "$partition" "$gres")"
done

all_eval_dependencies="${eval_general[0]}:${eval_general[1]}:${eval_general[2]}:${eval_useful[0]}:${eval_useful[1]}:${eval_useful[2]}:${eval_source[0]}:${eval_source[1]}:${eval_source[2]}:${eval_caps[0]}:${eval_caps[1]}:${eval_caps[2]}"
eval_validate="$(submit_job bonh_evalval_0918 eval_validate "$cpu" "$all_eval_dependencies" '' shared '' '')"
evalplus_prepare="$(submit_job bonh_eprep_0918 evalplus_prepare "$cpu" "${eval_caps[0]}:${eval_caps[1]}:${eval_caps[2]}" '' shared '' '')"
evalplus_run="$(submit_job bonh_eprun_0918 evalplus_run "$cpu" "$evalplus_prepare" '0-191%40' shared '' '')"
evalplus_aggregate="$(submit_job bonh_epagg_0918 evalplus_aggregate "$cpu" "$evalplus_run" '' shared '' '')"
report="$(submit_job bonh_report_0918 report "$cpu" "$eval_validate:$evalplus_aggregate:$weight_aggregate" '' shared '' '')"
audit="$(submit_job bonh_audit_0918 final_audit "$cpu" "$report" '' shared '' '')"
final_email="$(submit_job bonh_email_0918 final_email "$cpu" "$audit" '' shared '' '')"

printf 'experiment=pruning_bonham_20260918\ndry_run=%s\nresult_root=%s\nsubmission_log=%s\nfinal_audit_job=%s\nfinal_email_job=%s\n' \
  "$DRY_RUN" "$RESULT_ROOT" "$submission_log" "$audit" "$final_email" | tee -a "$submission_log"
