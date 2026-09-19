#!/usr/bin/env bash
set -Eeuo pipefail

: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${MODEL_KEY:?MODEL_KEY is required}"

mode="${1:?pipeline lane mode is required}"
shift
PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
cd "$REPO_DIR"

case "$mode" in
  random_and_states)
    if [[ ! -f "$RESULT_ROOT/masks/$MODEL_KEY/RANDOM_MASKS_COMPLETE.json" ]]; then
      "$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-random-mask \
        --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
    fi
    if [[ ! -f "$RESULT_ROOT/states/$MODEL_KEY/MASK_STATES_COMPLETE.json" ]]; then
      "$CPU_PYTHON_BIN" "$BUNDLE_DIR/campaign.py" build-mask-states \
        --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY"
    fi
    ;;
  steering)
    if [[ ! -f "$RESULT_ROOT/steering/$MODEL_KEY/fit/directions.COMPLETE.json" ]]; then
      "$PYTHON_BIN" "$BUNDLE_DIR/steering.py" extract \
        --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
    fi
    if [[ ! -f "$RESULT_ROOT/steering/$MODEL_KEY/frozen/COMPLETE.json" ]]; then
      "$PYTHON_BIN" "$BUNDLE_DIR/steering.py" develop \
        --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --hf-cache "$HF_CACHE_DIR"
    fi
    ;;
  evaluation)
    state_id="${1:?state id is required}"
    family_set="${2:?evaluation family set is required}"
    case "$family_set" in
      paper_core) families=(generalization useful_assertions source_attribution) ;;
      source_attribution) families=(source_attribution) ;;
      capabilities) families=(capabilities) ;;
      *) printf 'unknown evaluation family set: %s\n' "$family_set" >&2; exit 2 ;;
    esac
    "$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" run-state-sequence \
      --result-root "$RESULT_ROOT" --model-key "$MODEL_KEY" --state-id "$state_id" \
      --families "${families[@]}" --hf-cache "$HF_CACHE_DIR" \
      --batch-size "${EVALUATION_BATCH_SIZE:-4}"
    ;;
  *)
    printf 'unknown pipeline lane mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
