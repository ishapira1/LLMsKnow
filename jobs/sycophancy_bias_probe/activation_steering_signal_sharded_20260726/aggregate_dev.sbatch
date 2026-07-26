#!/bin/bash
#SBATCH --job-name=syco_sig_select_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/selection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/selection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726/common.sh
TASK_LABEL="pooled_signal_dev"
start_structured_task_log signal_select

mapfile -t INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/dev_screen/layer_*/question_results.jsonl' -type f | sort)
(( ${#INPUTS[@]} == 116 )) || {
  printf 'Expected 116 screen inputs, found %s\n' "${#INPUTS[@]}" >&2
  exit 1
}
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
  --output-dir "$INTERVENTION_ROOT/dev_aggregate"
  --n-bootstrap 500
  --seed 26
)
for input in "${INPUTS[@]}"; do
  cmd+=(--input "$input")
done
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
