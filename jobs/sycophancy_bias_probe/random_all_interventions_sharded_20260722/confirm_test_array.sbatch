#!/bin/bash
#SBATCH --job-name=syco_int_test_20260722
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/confirm/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/slurm/confirm/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/common.sh

mapfile -t SELECTED_INDICES < <(selected_base_indices)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#SELECTED_INDICES[@]} )); then
  printf '[confirm] skipping task_id=%s selected_count=%s\n' "$TASK_ID" "${#SELECTED_INDICES[@]}"
  exit 0
fi
resolve_task_context "${SELECTED_INDICES[$TASK_ID]}"
start_structured_task_log confirm_test probe_bias_random_all
validate_runtime_contract --require-cuda

test -s "$DIRECTIONS_PATH"
test -s "$SELECTION_JSON"
TEST_RANDOM_CONTROL_SEEDS="${TEST_RANDOM_CONTROL_SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_random_all_intervention.py" run-selected
  --source-run-dir "$SOURCE_RUN_DIR"
  --directions-path "$DIRECTIONS_PATH"
  --selection-json "$SELECTION_JSON"
  --output-root "$CELL_ROOT"
  --split test
  --random-control-seeds "$TEST_RANDOM_CONTROL_SEEDS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
print_command "${cmd[@]}"
"${cmd[@]}"

SELECTED_LAYER="$($ENV_PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_layer"]))' "$SELECTION_JSON")"
SHARD_DIR="$CELL_ROOT/layers/$(printf 'layer_%03d' "$SELECTED_LAYER")/test"
test -s "$SHARD_DIR/item_results_confirm.jsonl"
test -s "$SHARD_DIR/manifest_confirm.json"
printf '[confirm] completed selected_layer=%s shard_dir=%s\n' "$SELECTED_LAYER" "$SHARD_DIR"
