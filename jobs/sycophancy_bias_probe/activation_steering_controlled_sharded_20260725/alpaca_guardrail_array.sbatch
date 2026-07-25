#!/bin/bash
#SBATCH --job-name=syco_ctrl_alpaca_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/alpaca/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/alpaca/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_model "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
TASK_LABEL="${MODEL_KEY}_alpaca"
start_structured_task_log alpaca_guardrail
validate_runtime_contract --require-cuda
test -s "$ALPACA_UTILITY_MANIFEST"

SOURCE_RUN_DIR="$(source_run_dir "$MODEL_INDEX" 0)"
SELECTION="$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
LAYER="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=sys.argv[2]; rows=[r for r in p["selections"] if r["model_name"]==m]; assert len(rows)==1; print(rows[0]["selected_layer"])' \
  "$SELECTION" "$MODEL_IDENTIFIER"
)"
SELECTED_ALPHAS="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=sys.argv[2]; rows=[r for r in p["selections"] if r["model_name"]==m]; assert len(rows)==1; d=rows[0]["selected_dose"]["selected"]; values=sorted({-128.0,float(d["negative_alpha"]),0.0,float(d["positive_alpha"]),128.0}); print(",".join(str(value) for value in values))' \
  "$SELECTION" "$MODEL_IDENTIFIER"
)"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" run-alpaca-guardrail
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --alpaca-manifest "$ALPACA_UTILITY_MANIFEST"
  --directions-path "$DIRECTIONS_PATH"
  --output-dir "$MODEL_ROOT/alpaca_guardrail"
  --layer "$LAYER"
  --alphas="$SELECTED_ALPHAS"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
print_command "${cmd[@]}"
"${cmd[@]}"
