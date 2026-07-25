#!/bin/bash
#SBATCH --job-name=syco_ctrl_select_20260725
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/selection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/selection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
TASK_LABEL="pooled_dev"
start_structured_task_log aggregate_dev

mapfile -t INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/dev_screen/layer_*/question_results.jsonl' -type f | sort)
if (( ${#INPUTS[@]} == 0 )); then
  printf '%s\n' "No development screen outputs found." >&2
  exit 1
fi
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
  --output-dir "$INTERVENTION_ROOT/dev_aggregate"
  --n-bootstrap 2000
  --seed 5
)
for input in "${INPUTS[@]}"; do
  cmd+=(--input "$input")
done
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
