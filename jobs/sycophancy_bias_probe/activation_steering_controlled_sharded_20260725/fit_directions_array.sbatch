#!/bin/bash
#SBATCH --job-name=syco_ctrl_fit_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/fit/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/fit/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_model "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
TASK_LABEL="$MODEL_KEY"
start_structured_task_log fit_directions
validate_runtime_contract --require-cuda

SOURCE_CSQA="$(source_run_dir "$MODEL_INDEX" 0)"
SOURCE_ARC="$(source_run_dir "$MODEL_INDEX" 1)"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" fit-directions
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_CSQA"
  --source-run-dir "$SOURCE_ARC"
  --question-manifest "$QUESTION_MANIFEST"
  --output-dir "$DIRECTIONS_DIR"
  --control-seeds 0,1,2,3,4,5,6,7,8,9
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --progress-every 25
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$DIRECTIONS_PATH"
test -s "$DIRECTIONS_DIR/manifest.json"
