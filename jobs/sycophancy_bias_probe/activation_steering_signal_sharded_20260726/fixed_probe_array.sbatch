#!/bin/bash
#SBATCH --job-name=syco_sig_probe_20260726
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/probe/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/probe/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726/common.sh
resolve_cell "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
start_structured_task_log signal_probe
validate_runtime_contract --require-cuda

SELECTION="$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
LAYER="$(
  "$ENV_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); m=sys.argv[2]; rows=[r for r in p["selections"] if r["model_name"]==m]; assert len(rows)==1; print(rows[0]["selected_layer"])' \
  "$SELECTION" "$MODEL_IDENTIFIER"
)"
OUTPUT_DIR="$CELL_ROOT/fixed_probe_test/layer_$(printf '%03d' "$LAYER")"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" score-fixed-probe
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$QUESTION_MANIFEST"
  --directions-path "$DIRECTIONS_PATH"
  --output-dir "$OUTPUT_DIR"
  --split test
  --layers "$LAYER"
  --alphas=-4,0,4
  --control-seeds ""
  --learned-directions wn
  --max-batch-size 1
  --score-fixed-probe
  --no-generation-diagnostics
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
  --progress-every 5
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/question_results.jsonl"
