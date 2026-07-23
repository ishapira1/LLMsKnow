#!/bin/bash
#SBATCH --job-name=syco_int_loc_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --array=0-123
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/localize/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/localize/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

LAYERS_PER_MODEL=31
mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS=$(( ${#SELECTED_INDICES[@]} * LAYERS_PER_MODEL ))
if (( TASK_ID >= TOTAL_TASKS )); then
  printf '[localize] skipping task_id=%s total_tasks=%s\n' "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi
COMBO_OFFSET=$((TASK_ID / LAYERS_PER_MODEL))
LAYER=$((TASK_ID % LAYERS_PER_MODEL + 1))
resolve_task_context "${SELECTED_INDICES[$COMBO_OFFSET]}"
if (( LAYER > NONTERMINAL_LAYER_COUNT )); then
  printf '[localize] skipping terminal/out-of-range layer=%s task_label=%s nonterminal_layer_count=%s\n' \
    "$LAYER" "$TASK_LABEL" "$NONTERMINAL_LAYER_COUNT"
  exit 0
fi
start_structured_task_log "localize_layer_$(printf '%03d' "$LAYER")" probe_bias_random_all
validate_runtime_contract --require-cuda

test -s "$DIRECTIONS_PATH"
VAL_MAX_QUESTIONS="${VAL_MAX_QUESTIONS:-}"
RANDOM_CONTROL_SEEDS="${RANDOM_CONTROL_SEEDS:-0,1,2,3,4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" run-layer
  --source-run-dir "$SOURCE_RUN_DIR"
  --directions-path "$DIRECTIONS_PATH"
  --output-root "$CELL_ROOT"
  --layer "$LAYER"
  --split val
  --random-control-seeds "$RANDOM_CONTROL_SEEDS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --protocol patch-localize
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
if [[ -n "$VAL_MAX_QUESTIONS" ]]; then
  cmd+=(--max-questions "$VAL_MAX_QUESTIONS")
fi
print_command "${cmd[@]}"
"${cmd[@]}"

SHARD_DIR="$CELL_ROOT/layers/$(printf 'layer_%03d' "$LAYER")/val"
test -s "$SHARD_DIR/item_results_patch_localize.jsonl"
test -s "$SHARD_DIR/manifest_patch_localize.json"
test -s "$SHARD_DIR/baseline_replay_summary_patch_localize.json"
printf '[localize] completed layer=%s shard_dir=%s\n' "$LAYER" "$SHARD_DIR"
