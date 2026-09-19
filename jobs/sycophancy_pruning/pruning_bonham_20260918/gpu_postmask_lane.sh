#!/usr/bin/env bash
set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${MODEL_KEY:?MODEL_KEY is required}"

PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
cd "$REPO_DIR"
printf 'experiment=pruning_bonham_20260918\nstage=postmask_model_sequence\nmodel=%s\npython=%s\nhostname=%s\nstart_time=%s\n' \
  "$MODEL_KEY" "$PYTHON_BIN" "$(hostname)" "$(date -Is)"
nvidia-smi || true

if [[ ! -f "$RESULT_ROOT/masks/$MODEL_KEY/RANDOM_MASKS_COMPLETE.json" ]]; then
  command=(
    "$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-random-mask
    --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
  )
  print_command "${command[@]}"
  "${command[@]}"
else
  printf 'skip_stage=random_masks reason=authenticated_receipt_present\n'
fi

if [[ ! -f "$RESULT_ROOT/states/$MODEL_KEY/MASK_STATES_COMPLETE.json" ]]; then
  command=(
    "$CPU_PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-mask-states
    --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY"
  )
  print_command "${command[@]}"
  "${command[@]}"
else
  printf 'skip_stage=build_mask_states reason=authenticated_receipt_present\n'
fi

fit_receipt="$RESULT_ROOT/steering/$MODEL_KEY/fit/directions.COMPLETE.json"
if [[ ! -f "$fit_receipt" ]]; then
  command=(
    "$PYTHON_BIN" "$BUNDLE_DIR/steering.py" extract
    --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
  )
  print_command "${command[@]}"
  "${command[@]}"
else
  printf 'skip_stage=steering_extract reason=authenticated_receipt_present\n'
fi

develop_receipt="$RESULT_ROOT/steering/$MODEL_KEY/development/COMPLETE.json"
if [[ ! -f "$develop_receipt" ]]; then
  command=(
    "$PYTHON_BIN" "$BUNDLE_DIR/steering.py" develop
    --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
  )
  print_command "${command[@]}"
  "${command[@]}"
else
  printf 'skip_stage=steering_develop reason=authenticated_receipt_present\n'
fi

printf 'postmask_model_sequence_complete=1\nmodel=%s\nend_time=%s\n' \
  "$MODEL_KEY" "$(date -Is)"
