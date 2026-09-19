#!/usr/bin/env bash
set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${LANE_INDEX:?LANE_INDEX is required}"
: "${LANES:?LANES is required}"

if ! [[ "$LANE_INDEX" =~ ^[0-9]+$ && "$LANES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LANE_INDEX must be non-negative and LANES must be positive\n' >&2
  exit 2
fi
if (( LANE_INDEX >= LANES )); then
  printf 'LANE_INDEX must be smaller than LANES\n' >&2
  exit 2
fi

# The order balances manifest row counts under the two-lane accelerated setup:
# even slots contain four 512-row caches, while odd slots contain the 1,024-,
# 512-, and 256-row caches.  Cache identities do not depend on execution order.
PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
score_ids=(
  n1_seed5_prune
  selective_preserve
  n1_seed17_prune
  source_all_prune
  n1_seed29_prune
  source_false_prune
  general_preserve
)
blocks_per_pass="${BLOCKS_PER_PASS:-1}"
if ! [[ "$blocks_per_pass" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BLOCKS_PER_PASS must be a positive integer\n' >&2
  exit 2
fi

cd "$REPO_DIR"
printf 'score_lane=%s\nscore_lanes=%s\nmodel=%s\nblocks_per_pass=%s\nstart_time=%s\n' \
  "$LANE_INDEX" "$LANES" "$MODEL_KEY" "$blocks_per_pass" "$(date -Is)"
nvidia-smi || true
for ((index = LANE_INDEX; index < ${#score_ids[@]}; index += LANES)); do
  score_id="${score_ids[$index]}"
  command=(
    "$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" score-component
    --result-root "$RESULT_ROOT"
    --model-key "$MODEL_KEY"
    --score-id "$score_id"
    --hf-cache "$HF_CACHE_DIR"
    --blocks-per-pass "$blocks_per_pass"
  )
  printf 'score_index=%s\nscore_id=%s\n' "$index" "$score_id"
  print_command "${command[@]}"
  "${command[@]}"
done
printf 'score_lane_complete=1\nend_time=%s\n' "$(date -Is)"
