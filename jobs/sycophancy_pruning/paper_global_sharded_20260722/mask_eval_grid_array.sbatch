#!/bin/bash
#SBATCH --job-name=syco_prune_grid_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --array=0-9
#SBATCH --output=jobs/sycophancy_pruning/logs/paper_global_sharded_20260722/slurm/grid/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_pruning/logs/paper_global_sharded_20260722/slurm/grid/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
setup_environment

MANIFEST_SIZE="${MANIFEST_SIZE:-main}"
NSAMPLES="$(size_to_n "$MANIFEST_SIZE")"
GRID_Q="${GRID_Q:-0}"
GRID_EVAL_SPLITS="${GRID_EVAL_SPLITS:-val}"
CALIBRATION_SEED=5
P_VALUES=(0 1e-5 5e-5 7e-5 1e-4)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$GRID_Q" == "0" || "$GRID_Q" == "0.0" ]]; then
  if (( TASK_ID < 0 || TASK_ID >= 2 )); then
    printf '[grid] skipping baseline task_id=%s\n' "$TASK_ID"
    exit 0
  fi
  MODEL_INDEX="$TASK_ID"
  P_VALUE=0
else
  if (( TASK_ID < 0 || TASK_ID >= 10 )); then
    printf '[grid] skipping task_id=%s\n' "$TASK_ID"
    exit 0
  fi
  MODEL_INDEX=$((TASK_ID / 5))
  P_VALUE="${P_VALUES[$((TASK_ID % 5))]}"
fi
Q_VALUE="$GRID_Q"

MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
require_sha "MODEL_REVISION" "$MODEL_REVISION"
MODEL_SLUG="$(slugify "$MODEL_ID")"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"
PRUNE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/pruning.jsonl"
PRESERVE_MANIFEST="$BUNDLE_DIR/seed_${CALIBRATION_SEED}/${MANIFEST_SIZE}/preservation.jsonl"
EVALUATION_MANIFEST="$BUNDLE_DIR/evaluation/fixed_seed_5_heldout.jsonl"
require_file "$PRUNE_MANIFEST"
require_file "$PRESERVE_MANIFEST"
require_file "$EVALUATION_MANIFEST"
MANIFEST_IDENTITY="$(pruning_manifest_identity \
  "$PRUNE_MANIFEST" "$PRESERVE_MANIFEST" "$EVALUATION_MANIFEST")"

CONFIG_SLUG="p_$(numeric_slug "$P_VALUE")_q_$(numeric_slug "$Q_VALUE")"
CONFIG_ROOT="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/$CONFIG_SLUG"
PREDICTION_DIR="$PREDICTION_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/$CONFIG_SLUG/$GRID_EVAL_SPLITS"
POINTER_PATH="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/$CONFIG_SLUG.json"
STOP_PATH="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/p_$(numeric_slug "$P_VALUE")/hard_stop.json"
TASK_LABEL="${MODEL_SLUG}_${MANIFEST_SIZE}_${CONFIG_SLUG}"
STRUCTURED_DIR="jobs/sycophancy_pruning/logs/$BUNDLE_NAME/by_task/$MODEL_SLUG/grid_${MANIFEST_SIZE}/job_${SLURM_JOB_ID:-local}"
mkdir -p "$STRUCTURED_DIR" "$CONFIG_ROOT" "$PREDICTION_DIR" "$(dirname "$POINTER_PATH")"
exec > >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${TASK_ID}.err" >&2)

if [[ "$Q_VALUE" != "0" && -f "$STOP_PATH" ]]; then
  SHOULD_STOP="$($ENV_PYTHON -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["stop"] else "0")' "$STOP_PATH")"
  if [[ "$SHOULD_STOP" == "1" ]]; then
    mkdir -p "$CONFIG_ROOT"
    printf '{"status":"skipped_after_hard_utility_stop","manifest_identity":"%s","sentinel":"%s","p":"%s","q":"%s"}\n' \
      "$MANIFEST_IDENTITY" "$STOP_PATH" "$P_VALUE" "$Q_VALUE" > "$CONFIG_ROOT/skipped.json"
    print_task_start "$TASK_LABEL" "hard utility stop precheck"
    printf '[grid] skipped model=%s p=%s q=%s due to %s\n' "$MODEL_ID" "$P_VALUE" "$Q_VALUE" "$STOP_PATH"
    print_task_end "$TASK_LABEL" 0 0
    exit 0
  fi
fi

prune_cmd=(
  "$ENV_PYTHON" "$HARM_REPO_DIR/src/prune.py"
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --tokenizer_revision "$MODEL_REVISION"
  --prune_method attribution_score_set_difference_global
  --prune_manifest "$PRUNE_MANIFEST"
  --preserve_manifest "$PRESERVE_MANIFEST"
  --nsamples "$NSAMPLES"
  --nsamples_preserve "$NSAMPLES"
  --seed "$CALIBRATION_SEED"
  --score_format raw
  --loss_mode completion_nll
  --attribution_variant paper
  --artifact_root "$PRUNING_ARTIFACT_ROOT"
  --p "$P_VALUE"
  --q "$Q_VALUE"
  --control none
  --no_abs
  --neg_prune
  --abs_preserve
  --dump_mask
)
if [[ "$Q_VALUE" != "0" ]]; then
  prune_cmd+=(--use_saved_scores)
fi
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
printf '[grid] model=%s revision=%s size=%s manifest_identity=%s p=%s q=%s predictions=%s\n' \
  "$MODEL_ID" "$MODEL_REVISION" "$MANIFEST_SIZE" "$MANIFEST_IDENTITY" \
  "$P_VALUE" "$Q_VALUE" "$PREDICTION_DIR"
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
  --harm-repo "$HARM_REPO_DIR"
  --artifact-root "$PRUNING_ARTIFACT_ROOT"
  --model "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --prune-manifest "$PRUNE_MANIFEST"
  --preserve-manifest "$PRESERVE_MANIFEST"
  --nsamples "$NSAMPLES"
  --nsamples-preserve "$NSAMPLES"
  --seed "$CALIBRATION_SEED"
  --score-format raw
  --loss-mode completion_nll
  --attribution-variant paper
  --p "$P_VALUE"
  --q "$Q_VALUE"
  --neg-prune
  --abs-preserve
  --control none
  --require-existing
  --output "$POINTER_PATH"
)
"${resolver_cmd[@]}"
METADATA_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["metadata_path"])' "$POINTER_PATH")"
EVALUATION_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_path"])' "$POINTER_PATH")"
INDICES_PATH="$($ENV_PYTHON -c 'import json,sys; value=json.load(open(sys.argv[1]))["indices_path"]; print(value or "")' "$POINTER_PATH")"
PRESERVATION_LOSS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["preservation_loss"])' "$EVALUATION_PATH")"
WIKITEXT_PPL="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["wikitext_perplexity"])' "$EVALUATION_PATH")"

inference_cmd=(
  "$ENV_PYTHON" scripts/run_pruning_heldout_inference.py
  --evaluation-manifest "$EVALUATION_MANIFEST"
  --output-dir "$PREDICTION_DIR"
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
  --splits "$GRID_EVAL_SPLITS"
  --overwrite
)
if [[ -n "$INDICES_PATH" ]]; then
  inference_cmd+=(--indices-path "$INDICES_PATH" --mask-metadata "$METADATA_PATH")
fi
if [[ "${DEVICE_MAP_AUTO:-0}" == "1" ]]; then
  inference_cmd+=(--device-map-auto)
fi
"${inference_cmd[@]}"

if [[ "$Q_VALUE" == "0" ]]; then
  # The live candidate is the canonical replayed base checkpoint. The historical
  # baseline emitted beside it remains available as a reproducibility audit.
  "$ENV_PYTHON" scripts/aggregate_pruning_offline_evaluation.py \
    --baseline "$PREDICTION_DIR/baseline_items.jsonl" \
    --candidate "$PREDICTION_DIR/candidate_items.jsonl" \
    --output-dir "$CONFIG_ROOT/baseline_replay_$GRID_EVAL_SPLITS" \
    --p 0 --q 0 --calibration-seed 5 --actual-mask-count 0 \
    --baseline-preservation-loss "$PRESERVATION_LOSS" \
    --candidate-preservation-loss "$PRESERVATION_LOSS" \
    --baseline-wikitext-perplexity "$WIKITEXT_PPL" \
    --candidate-wikitext-perplexity "$WIKITEXT_PPL" \
    --baseline-evaluation-artifact "$EVALUATION_PATH" \
    --candidate-evaluation-artifact "$EVALUATION_PATH" \
    --n-bootstrap "${GRID_BOOTSTRAP_SAMPLES:-0}"
else
  BASE_CONFIG_ROOT="$ANALYSIS_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/p_0_q_0"
  BASE_PREDICTION_DIR="$PREDICTION_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/p_0_q_0/$GRID_EVAL_SPLITS"
  BASE_POINTER="$REGISTRY_ROOT/$MODEL_SLUG/revision_$(slugify "$MODEL_REVISION")/seed_5/$MANIFEST_SIZE/primary/$MANIFEST_IDENTITY/p_0_q_0.json"
  require_file "$BASE_PREDICTION_DIR/candidate_items.jsonl"
  require_file "$BASE_POINTER"
  BASE_EVALUATION_PATH="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_path"])' "$BASE_POINTER")"
  BASE_PRESERVATION_LOSS="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["preservation_loss"])' "$BASE_EVALUATION_PATH")"
  BASE_WIKITEXT_PPL="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["wikitext_perplexity"])' "$BASE_EVALUATION_PATH")"
  ACTUAL_MASK_COUNT="$($ENV_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["surviving_count"])' "$METADATA_PATH")"

  "$ENV_PYTHON" scripts/aggregate_pruning_offline_evaluation.py \
    --baseline "$BASE_PREDICTION_DIR/candidate_items.jsonl" \
    --candidate "$PREDICTION_DIR/candidate_items.jsonl" \
    --output-dir "$CONFIG_ROOT/evaluation" \
    --p "$P_VALUE" --q "$Q_VALUE" --calibration-seed 5 \
    --actual-mask-count "$ACTUAL_MASK_COUNT" \
    --baseline-preservation-loss "$BASE_PRESERVATION_LOSS" \
    --candidate-preservation-loss "$PRESERVATION_LOSS" \
    --baseline-wikitext-perplexity "$BASE_WIKITEXT_PPL" \
    --candidate-wikitext-perplexity "$WIKITEXT_PPL" \
    --baseline-evaluation-artifact "$BASE_EVALUATION_PATH" \
    --candidate-evaluation-artifact "$EVALUATION_PATH" \
    --n-bootstrap "${GRID_BOOTSTRAP_SAMPLES:-0}"

  "$ENV_PYTHON" scripts/check_pruning_hard_stop.py \
    --selection-summary "$CONFIG_ROOT/evaluation/selection_summary.csv" \
    --output "$STOP_PATH" \
    --max-neutral-accuracy-drop "${HARD_NEUTRAL_ACCURACY_DROP:-0.02}" \
    --max-neutral-probability-drop "${HARD_NEUTRAL_PROBABILITY_DROP:-0.02}" \
    --max-correction-accuracy-drop "${HARD_CORRECTION_ACCURACY_DROP:-0.02}" \
    --max-agreement-accuracy-drop "${HARD_AGREEMENT_ACCURACY_DROP:-0.02}" \
    --max-preservation-loss-increase "${HARD_PRESERVATION_LOSS_INCREASE:-0.05}" \
    --max-wikitext-perplexity-increase "${HARD_WIKITEXT_PPL_INCREASE:-0.05}" \
    --max-other-wrong-invalid-increase "${HARD_OTHER_WRONG_INVALID_INCREASE:-0.02}"
fi

END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" 0 "$((END_EPOCH - START_EPOCH))"
