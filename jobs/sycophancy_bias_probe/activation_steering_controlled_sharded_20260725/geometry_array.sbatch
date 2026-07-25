#!/bin/bash
#SBATCH --job-name=syco_ctrl_geom_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/geometry/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/geometry/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_cell "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
start_structured_task_log geometry
validate_runtime_contract --require-cuda

SELECTION="$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
LAYERS="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=sys.argv[2]; rows=[r for r in p["selections"] if r["model_name"]==m]; assert len(rows)==1; print(",".join(str(x) for x in rows[0]["test_layers"]))' \
  "$SELECTION" "$MODEL_IDENTIFIER"
)"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-geometry
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$QUESTION_MANIFEST"
  --directions-path "$DIRECTIONS_PATH"
  --output-dir "$CELL_ROOT/geometry_test"
  --split test
  --layers "$LAYERS"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
print_command "${cmd[@]}"
"${cmd[@]}"
