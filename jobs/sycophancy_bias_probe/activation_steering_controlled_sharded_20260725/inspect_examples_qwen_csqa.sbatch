#!/bin/bash
#SBATCH --job-name=syco_ctrl_inspect_20260725
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/inspection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/inspection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
QUESTION_MANIFEST="${QUESTION_MANIFEST:-configs/experiments/activation_steering_preflight_8_20260725.jsonl}"
export QUESTION_MANIFEST
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_cell 1
start_structured_task_log inspect_examples
validate_runtime_contract --require-cuda

LAYERS="${LAYERS:-17,18}"
OUTPUT_DIR="$INTERVENTION_ROOT/preflight_inspection/qwen25_7b/commonsense_qa"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" inspect-examples
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$PREFLIGHT_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --device cuda
  --hf-cache-dir "$HF_CACHE_DIR"
  --torch-dtype auto
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/preflight_examples.jsonl"
test -s "$OUTPUT_DIR/manifest.json"
