#!/bin/bash
#SBATCH --job-name=syco_ctrl_valid_20260725
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/validation/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/validation/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
resolve_model "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
TASK_LABEL="$MODEL_KEY"
start_structured_task_log validate_source

SOURCE_CSQA="$(source_run_dir "$MODEL_INDEX" 0)"
SOURCE_ARC="$(source_run_dir "$MODEL_INDEX" 1)"
OUTPUT_DIR="$INTERVENTION_ROOT/source_validation/$MODEL_KEY"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" validate-source
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_CSQA"
  --source-run-dir "$SOURCE_ARC"
  --question-manifest "$QUESTION_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --require-human-approval
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/source_validation.json"
