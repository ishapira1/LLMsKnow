#!/bin/bash
#SBATCH --job-name=syco_ctrl_screen_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-115
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/screen/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/screen/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_layer_task "${SLURM_ARRAY_TASK_ID:-0}" || exit 0
model_selected "$MODEL_INDEX" || exit 0
start_structured_task_log screen_layer
validate_runtime_contract --require-cuda

OUTPUT_DIR="$CELL_ROOT/dev_screen/layer_$(printf '%03d' "$LAYER")"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" screen-layers
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$QUESTION_MANIFEST"
  --directions-path "$DIRECTIONS_PATH"
  --output-dir "$OUTPUT_DIR"
  --split val
  --layers "$LAYER"
  --alphas=-32,-8,-2,0,2,8,32
  --control-seeds 0,1,2,3,4,5,6,7,8,9
  --learned-directions wn,cn,wc,sw
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
