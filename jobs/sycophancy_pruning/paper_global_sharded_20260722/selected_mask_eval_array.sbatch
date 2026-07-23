#!/bin/bash
#SBATCH --job-name=weight_pruning_selected
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --array=0-17
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/selected/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/selected/%x.%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_gpu_allocation
setup_environment

MANIFEST_SIZE="${MANIFEST_SIZE:-main}"
NSAMPLES="$(size_to_n "$MANIFEST_SIZE")"
SELECTED_VARIANTS_CSV="${SELECTED_VARIANTS_CSV:-primary,structure_matched,opposite_sign,second_slice,random_magnitude,alpaca_only,chat,choice_token,released_abs}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"
IFS=, read -r -a SELECTED_VARIANTS <<< "$SELECTED_VARIANTS_CSV"
N_VARIANTS="${#SELECTED_VARIANTS[@]}"
TASK_ID="$SLURM_ARRAY_TASK_ID"
TOTAL_TASKS=$((2 * N_VARIANTS))
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  printf '[selected] skipping task_id=%s total_tasks=%s\n' "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi
MODEL_INDEX=$((TASK_ID / N_VARIANTS))
VARIANT_INDEX=$((TASK_ID % N_VARIANTS))
VARIANT="${SELECTED_VARIANTS[$VARIANT_INDEX]}"
MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
require_sha "MODEL_REVISION" "$MODEL_REVISION"
MODEL_SLUG="$(slugify "$MODEL_ID")"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"
PRIMARY_PRUNE_MANIFEST="$BUNDLE_DIR/seed_5/${MANIFEST_SIZE}/pruning.jsonl"
PRIMARY_PRESERVE_MANIFEST="$BUNDLE_DIR/seed_5/${MANIFEST_SIZE}/preservation.jsonl"
EVALUATION_MANIFEST="$BUNDLE_DIR/evaluation/fixed_seed_5_heldout.jsonl"
PRIMARY_MANIFEST_IDENTITY="$(pruning_manifest_identity \
  "$PRIMARY_PRUNE_MANIFEST" "$PRIMARY_PRESERVE_MANIFEST" "$EVALUATION_MANIFEST")"
VARIANT_SLUG="$(slugify "$VARIANT")"
TASK_LABEL="${MODEL_SLUG}_${MANIFEST_SIZE}_${VARIANT_SLUG}"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/selected_${VARIANT_SLUG}/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.err" >&2)
SELECTION_JSON="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/selection/$PRIMARY_MANIFEST_IDENTITY/selected_configuration.json"
require_file "$SELECTION_JSON"
SELECTION_STATUS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["status"])' "$SELECTION_JSON")"
if [[ "$SELECTION_STATUS" != "selected" ]]; then
  print_task_start "$TASK_LABEL" "no feasible mask; retain base model"
  printf '[selected] model=%s has no feasible mask; retaining base model and skipping variant=%s\n' "$MODEL_ID" "$VARIANT"
  print_task_end "$TASK_LABEL" 0 0
  exit 0
fi
P_VALUE="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["selected_p"])' "$SELECTION_JSON")"
Q_VALUE="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["selected_q"])' "$SELECTION_JSON")"

CALIBRATION_SEED=5
SCORE_VARIANT=primary
SCORE_FORMAT=raw
LOSS_MODE=completion_nll
ATTRIBUTION_VARIANT=paper
CONTROL=none
NEG_PRUNE=1
SECOND_SLICE=0
case "$VARIANT" in
  primary) ;;
  primary_seed17) CALIBRATION_SEED=17 ;;
  primary_seed29) CALIBRATION_SEED=29 ;;
  structure_matched) SCORE_VARIANT=structure_matched; CONTROL=structure_matched ;;
  opposite_sign) NEG_PRUNE=0 ;;
  second_slice) SECOND_SLICE=1 ;;
  random_magnitude) CONTROL=random_magnitude ;;
  alpaca_only) SCORE_VARIANT=alpaca_only; CONTROL=alpaca_only ;;
  chat) SCORE_VARIANT=chat; SCORE_FORMAT=chat ;;
  choice_token) SCORE_VARIANT=choice_token; LOSS_MODE=choice_token ;;
  released_abs) SCORE_VARIANT=released_abs; ATTRIBUTION_VARIANT=released_abs; NEG_PRUNE=0 ;;
  *) printf '%s\n' "Unknown selected variant: $VARIANT" >&2; exit 2 ;;
esac

PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/pruning.jsonl"
PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/preservation.jsonl"
if [[ "$SCORE_VARIANT" == "structure_matched" ]]; then
  PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/structure_control.jsonl"
elif [[ "$SCORE_VARIANT" == "alpaca_only" ]]; then
  PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/alpaca_preservation.jsonl"
fi
require_file "$PRUNE_MANIFEST"
require_file "$PRESERVE_MANIFEST"
require_file "$EVALUATION_MANIFEST"
MANIFEST_IDENTITY="$(pruning_manifest_identity \
  "$PRUNE_MANIFEST" "$PRESERVE_MANIFEST" "$EVALUATION_MANIFEST")"

# Preservation-loss baselines must use the same seed, manifest, format, and
# loss contract as the candidate sensitivity. Attribution/control direction
# does not affect the unmodified q=0 checkpoint.
BASE_PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/pruning.jsonl"
BASE_PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/preservation.jsonl"
BASE_SCORE_FORMAT="$SCORE_FORMAT"
BASE_LOSS_MODE="$LOSS_MODE"
BASELINE_VARIANT="primary"
if [[ "$SCORE_VARIANT" == "alpaca_only" ]]; then
  BASE_PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/alpaca_preservation.jsonl"
  BASELINE_VARIANT=alpaca_only
elif [[ "$SCORE_FORMAT" == "chat" ]]; then
  BASELINE_VARIANT=chat
elif [[ "$LOSS_MODE" == "choice_token" ]]; then
  BASELINE_VARIANT=choice_token
fi
BASELINE_MANIFEST_IDENTITY="$(pruning_manifest_identity \
  "$BASE_PRUNE_MANIFEST" "$BASE_PRESERVE_MANIFEST" "$EVALUATION_MANIFEST")"

CONFIG_SLUG="p_$(numeric_slug "$P_VALUE")_q_$(numeric_slug "$Q_VALUE")"
CONFIG_ROOT="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_${CALIBRATION_SEED}/$MANIFEST_SIZE/selected/$VARIANT_SLUG/$MANIFEST_IDENTITY/$CONFIG_SLUG"
PREDICTION_DIR="$PREDICTION_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_${CALIBRATION_SEED}/$MANIFEST_SIZE/selected/$VARIANT_SLUG/$MANIFEST_IDENTITY/$CONFIG_SLUG/$EVAL_SPLITS"
POINTER_PATH="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_${CALIBRATION_SEED}/$MANIFEST_SIZE/selected/$VARIANT_SLUG/$MANIFEST_IDENTITY/$CONFIG_SLUG.json"
mkdir -p "$CONFIG_ROOT" "$PREDICTION_DIR" "$(dirname "$POINTER_PATH")"

BASELINE_POINTER="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_${CALIBRATION_SEED}/$MANIFEST_SIZE/selected_baselines/$BASELINE_VARIANT/$BASELINE_MANIFEST_IDENTITY/p_0_q_0.json"
mkdir -p "$(dirname "$BASELINE_POINTER")"
if ! command -v flock >/dev/null 2>&1; then
  printf '%s\n' "flock is required to serialize shared q=0 sensitivity baselines" >&2
  exit 1
fi
exec 9>"${BASELINE_POINTER}.lock"
flock 9
baseline_resolver=(
  "$ENV_PYTHON" scripts/resolve_paper_pruning_artifact.py
  --artifact-root "$PRUNING_ARTIFACT_ROOT"
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --prune-manifest "$BASE_PRUNE_MANIFEST"
  --preserve-manifest "$BASE_PRESERVE_MANIFEST"
  --nsamples "$NSAMPLES"
  --nsamples-preserve "$NSAMPLES"
  --seed "$CALIBRATION_SEED"
  --score-format "$BASE_SCORE_FORMAT"
  --loss-mode "$BASE_LOSS_MODE"
  --attribution-variant paper
  --p 0 --q 0
  --neg-prune --abs-preserve --control none
)
BASE_EVALUATION_PATH="$("${baseline_resolver[@]}" --print-field evaluation_path)"
if [[ ! -f "$BASE_EVALUATION_PATH" ]]; then
  baseline_cmd=(
    "$ENV_PYTHON" "$WEIGHT_PRUNING_DIR/prune.py"
    --model "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --tokenizer_revision "$MODEL_REVISION"
    --prune_method attribution_score_set_difference_global
    --prune_manifest "$BASE_PRUNE_MANIFEST"
    --preserve_manifest "$BASE_PRESERVE_MANIFEST"
    --nsamples "$NSAMPLES"
    --nsamples_preserve "$NSAMPLES"
    --seed "$CALIBRATION_SEED"
    --score_format "$BASE_SCORE_FORMAT"
    --loss_mode "$BASE_LOSS_MODE"
    --attribution_variant paper
    --artifact_root "$PRUNING_ARTIFACT_ROOT"
    --p 0 --q 0 --control none
    --no_abs --neg_prune --abs_preserve --dump_mask
  )
  printf '[selected] creating matched q=0 preservation baseline variant=%s seed=%s\n' \
    "$BASELINE_VARIANT" "$CALIBRATION_SEED"
  "${baseline_cmd[@]}"
fi
"${baseline_resolver[@]}" --require-existing --output "$BASELINE_POINTER"
flock -u 9

prune_cmd=(
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
  --p "$P_VALUE"
  --q "$Q_VALUE"
  --control "$CONTROL"
  --no_abs
  --abs_preserve
  --use_saved_scores
  --dump_mask
)
if [[ "$NEG_PRUNE" == "1" ]]; then prune_cmd+=(--neg_prune); fi
if [[ "$SECOND_SLICE" == "1" ]]; then prune_cmd+=(--freeze_first_top_q); fi
if [[ "${SAVE_SELECTED_MODEL:-0}" == "1" ]]; then prune_cmd+=(--save_model); fi
if [[ "${RUN_ZERO_SHOT:-0}" == "1" ]]; then prune_cmd+=(--eval_zero_shot); fi
if [[ "${RUN_ALPACA_EVAL:-0}" == "1" ]]; then
  ALPACA_UTILITY_MANIFEST="$BUNDLE_DIR/evaluation/alpaca_utility.jsonl"
  require_file "$ALPACA_UTILITY_MANIFEST"
  prune_cmd+=(
    --eval_alpaca
    --alpaca_eval_data "$ALPACA_UTILITY_MANIFEST"
    --alpaca_eval_seed "${ALPACA_EVAL_SEED:-5}"
    --alpaca_nsamples "${ALPACA_EVAL_NSAMPLES:-1000}"
  )
fi

printf -v command_text '%q ' "${prune_cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[selected] model=%s seed=%s variant=%s manifest_identity=%s p=%s q=%s splits=%s\n' \
  "$MODEL_ID" "$CALIBRATION_SEED" "$VARIANT" "$MANIFEST_IDENTITY" \
  "$P_VALUE" "$Q_VALUE" "$EVAL_SPLITS"
START_EPOCH="$(date +%s)"
set +e
"${prune_cmd[@]}"
STATUS="$?"
set -e
if [[ "$STATUS" -ne 0 ]]; then
  END_EPOCH="$(date +%s)"
  print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
  exit "$STATUS"
fi

resolver_cmd=(
  "$ENV_PYTHON" scripts/resolve_paper_pruning_artifact.py
  --artifact-root "$PRUNING_ARTIFACT_ROOT"
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --prune-manifest "$PRUNE_MANIFEST"
  --preserve-manifest "$PRESERVE_MANIFEST"
  --nsamples "$NSAMPLES"
  --nsamples-preserve "$NSAMPLES"
  --seed "$CALIBRATION_SEED"
  --score-format "$SCORE_FORMAT"
  --loss-mode "$LOSS_MODE"
  --attribution-variant "$ATTRIBUTION_VARIANT"
  --p "$P_VALUE"
  --q "$Q_VALUE"
  --control "$CONTROL"
  --abs-preserve
  --require-existing
  --output "$POINTER_PATH"
)
if [[ "$NEG_PRUNE" == "1" ]]; then resolver_cmd+=(--neg-prune); else resolver_cmd+=(--no-neg-prune); fi
if [[ "$SECOND_SLICE" == "1" ]]; then resolver_cmd+=(--freeze-first-top-q); fi
"${resolver_cmd[@]}"

METADATA_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata_path"])' "$POINTER_PATH")"
EVALUATION_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_path"])' "$POINTER_PATH")"
INDICES_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["indices_path"])' "$POINTER_PATH")"
PRESERVATION_LOSS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["preservation_loss"])' "$EVALUATION_PATH")"
WIKITEXT_PPL="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["wikitext_perplexity"])' "$EVALUATION_PATH")"
ACTUAL_MASK_COUNT="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["surviving_count"])' "$METADATA_PATH")"

inference_cmd=(
  "$ENV_PYTHON" scripts/run_pruning_heldout_inference.py
  --evaluation-manifest "$EVALUATION_MANIFEST"
  --output-dir "$PREDICTION_DIR"
  --indices-path "$INDICES_PATH"
  --mask-metadata "$METADATA_PATH"
  --expected-mask-count "$ACTUAL_MASK_COUNT"
  --p "$P_VALUE"
  --q "$Q_VALUE"
  --calibration-seed "$CALIBRATION_SEED"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-32}"
  --generation-seed 5
  --preservation-loss "$PRESERVATION_LOSS"
  --wikitext-perplexity "$WIKITEXT_PPL"
  --splits "$EVAL_SPLITS"
  --overwrite
)
if [[ "${DEVICE_MAP_AUTO:-0}" == "1" ]]; then inference_cmd+=(--device-map-auto); fi
"${inference_cmd[@]}"

BASE_PREDICTION_DIR="$PREDICTION_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$PRIMARY_MANIFEST_IDENTITY/p_0_q_0/$EVAL_SPLITS"
require_file "$BASE_PREDICTION_DIR/candidate_items.jsonl"
BASE_PRESERVATION_LOSS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["preservation_loss"])' "$BASE_EVALUATION_PATH")"
BASE_WIKITEXT_PPL="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["wikitext_perplexity"])' "$BASE_EVALUATION_PATH")"

"$ENV_PYTHON" scripts/aggregate_pruning_offline_evaluation.py \
  --baseline "$BASE_PREDICTION_DIR/candidate_items.jsonl" \
  --candidate "$PREDICTION_DIR/candidate_items.jsonl" \
  --output-dir "$CONFIG_ROOT/evaluation_$EVAL_SPLITS" \
  --p "$P_VALUE" --q "$Q_VALUE" --calibration-seed "$CALIBRATION_SEED" \
  --actual-mask-count "$ACTUAL_MASK_COUNT" \
  --baseline-preservation-loss "$BASE_PRESERVATION_LOSS" \
  --candidate-preservation-loss "$PRESERVATION_LOSS" \
  --baseline-wikitext-perplexity "$BASE_WIKITEXT_PPL" \
  --candidate-wikitext-perplexity "$WIKITEXT_PPL" \
  --baseline-evaluation-artifact "$BASE_EVALUATION_PATH" \
  --candidate-evaluation-artifact "$EVALUATION_PATH" \
  --n-bootstrap "${SELECTED_BOOTSTRAP_SAMPLES:-2000}" \
  --bootstrap-seed 5

END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" 0 "$((END_EPOCH - START_EPOCH))"
