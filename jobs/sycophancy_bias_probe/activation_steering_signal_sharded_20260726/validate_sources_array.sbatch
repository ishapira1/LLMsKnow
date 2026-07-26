#!/bin/bash
#SBATCH --job-name=syco_sig_validate_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/validation/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/validation/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726/common.sh
resolve_cell "${SLURM_ARRAY_TASK_ID:-0}"
model_selected "$MODEL_INDEX" || exit 0
start_structured_task_log signal_validate

OUTPUT_DIR="$CELL_ROOT/source_validation"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" validate-source
  --config "$CONFIG_PATH"
  --source-run-dir "$SOURCE_RUN_DIR"
  --question-manifest "$QUESTION_MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --no-require-human-approval
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/source_validation.json"
