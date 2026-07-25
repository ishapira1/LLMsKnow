#!/bin/bash
#SBATCH --job-name=syco_ctrl_xfer_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/transfer/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/transfer/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_cell "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
start_structured_task_log cross_dataset_transfer
validate_runtime_contract --require-cuda

SOURCE_DATASET_INDEX=$((1 - DATASET_INDEX))
SOURCE_DATASET_NAME="${DATASET_NAMES[$SOURCE_DATASET_INDEX]}"
TRANSFER_DIRECTIONS="$INTERVENTION_ROOT/directions_dataset/$MODEL_KEY/$SOURCE_DATASET_NAME/directions.npz"
SELECTION="$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
LAYER="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=sys.argv[2]; rows=[r for r in p["selections"] if r["model_name"]==m]; assert len(rows)==1; print(rows[0]["selected_layer"])' \
  "$SELECTION" "$MODEL_IDENTIFIER"
)"
OUTPUT_DIR="$CELL_ROOT/transfer_from_${SOURCE_DATASET_NAME}/layer_$(printf '%03d' "$LAYER")"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-selected
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$QUESTION_MANIFEST"
  --directions-path "$TRANSFER_DIRECTIONS"
  --output-dir "$OUTPUT_DIR"
  --split test
  --layers "$LAYER"
  --control-seeds 0,1,2,3,4,5,6,7,8,9
  --learned-directions wn
  --max-batch-size 1
  --no-score-fixed-probe
  --generation-diagnostics
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --progress-every 10
)
print_command "${cmd[@]}"
"${cmd[@]}"
