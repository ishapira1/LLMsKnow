#!/bin/bash
#SBATCH --job-name=weight_pruning_score
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/scores/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/scores/%x.%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_gpu_allocation
setup_environment

MANIFEST_SIZE="${MANIFEST_SIZE:-smoke}"
NSAMPLES="$(size_to_n "$MANIFEST_SIZE")"
SCORE_VARIANT="${SCORE_VARIANT:-primary}"
CALIBRATION_SEEDS_CSV="${CALIBRATION_SEEDS_CSV:-5}"
IFS=, read -r -a CALIBRATION_SEEDS <<< "$CALIBRATION_SEEDS_CSV"
N_SEEDS="${#CALIBRATION_SEEDS[@]}"
if (( N_SEEDS == 0 )); then
  printf '%s\n' "CALIBRATION_SEEDS_CSV resolved to no seeds" >&2
  exit 2
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"
TOTAL_TASKS=$((2 * N_SEEDS * 2))
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  printf '[score] skipping task_id=%s total_tasks=%s\n' "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi
ROLE_INDEX=$((TASK_ID % 2))
SEED_INDEX=$(((TASK_ID / 2) % N_SEEDS))
MODEL_INDEX=$((TASK_ID / (2 * N_SEEDS)))
if (( ROLE_INDEX == 0 )); then ROLE=prune; else ROLE=preserve; fi
CALIBRATION_SEED="${CALIBRATION_SEEDS[$SEED_INDEX]}"
MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
require_sha "MODEL_REVISION" "$MODEL_REVISION"
MODEL_SLUG="$(slugify "$MODEL_ID")"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"

PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/pruning.jsonl"
PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/preservation.jsonl"
SCORE_FORMAT=raw
LOSS_MODE=completion_nll
ATTRIBUTION_VARIANT=paper
CONTROL=none
case "$SCORE_VARIANT" in
  primary) ;;
  structure_matched)
    PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/structure_control.jsonl"
    CONTROL=structure_matched
    ;;
  alpaca_only)
    PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/alpaca_preservation.jsonl"
    CONTROL=alpaca_only
    ;;
  chat)
    SCORE_FORMAT=chat
    ;;
  choice_token)
    LOSS_MODE=choice_token
    ;;
  released_abs)
    ATTRIBUTION_VARIANT=released_abs
    ;;
  *)
    printf '%s\n' "Unknown SCORE_VARIANT=$SCORE_VARIANT" >&2
    exit 2
    ;;
esac
require_file "$PRUNE_MANIFEST"
require_file "$PRESERVE_MANIFEST"

TASK_LABEL="${MODEL_SLUG}_seed${CALIBRATION_SEED}_${MANIFEST_SIZE}_${SCORE_VARIANT}_${ROLE}"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/scores_${SCORE_VARIANT}_${MANIFEST_SIZE}/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.err" >&2)

cmd=(
  "$ENV_PYTHON" "$WEIGHT_PRUNING_DIR/prune.py"
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --tokenizer_revision "$MODEL_REVISION"
  --prune_method attribution_score_set_difference_global
  --prune_manifest "$PRUNE_MANIFEST"
  --preserve_manifest "$PRESERVE_MANIFEST"
  --nsamples "$NSAMPLES"
  --nsamples_preserve "$NSAMPLES"
  --seed "$CALIBRATION_SEED"
  --score_format "$SCORE_FORMAT"
  --loss_mode "$LOSS_MODE"
  --attribution_variant "$ATTRIBUTION_VARIANT"
  --artifact_root "$PRUNING_ARTIFACT_ROOT"
  --score_role "$ROLE"
  --control "$CONTROL"
  --no_abs
  --neg_prune
  --abs_preserve
  --dump_score
)

printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[score] model=%s revision=%s seed=%s size=%s role=%s variant=%s prune_manifest=%s preserve_manifest=%s\n' \
  "$MODEL_ID" "$MODEL_REVISION" "$CALIBRATION_SEED" "$MANIFEST_SIZE" "$ROLE" "$SCORE_VARIANT" \
  "$PRUNE_MANIFEST" "$PRESERVE_MANIFEST"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
