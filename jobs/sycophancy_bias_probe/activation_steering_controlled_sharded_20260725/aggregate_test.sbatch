#!/bin/bash
#SBATCH --job-name=syco_ctrl_agg_20260725
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/aggregate/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_controlled_sharded_20260725/slurm/aggregate/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/common.sh
TASK_LABEL="pooled_test"
start_structured_task_log aggregate_test
mapfile -t INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/test_selected/layer_*/question_results.jsonl' -type f | sort)
if (( ${#INPUTS[@]} == 0 )); then
  printf '%s\n' "No selected test outputs found." >&2
  exit 1
fi
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
  --output-dir "$INTERVENTION_ROOT/test_aggregate"
  --n-bootstrap 2000
  --seed 5
)
for input in "${INPUTS[@]}"; do
  cmd+=(--input "$input")
done
print_command "${cmd[@]}"
"${cmd[@]}"

mapfile -t DOSE_INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/dev_selected/layer_*/question_results.jsonl' -type f | sort)
if (( ${#DOSE_INPUTS[@]} > 0 )); then
  dose_cmd=(
    "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
    --output-dir "$INTERVENTION_ROOT/dev_selected_aggregate"
    --n-bootstrap 2000
    --seed 5
  )
  for input in "${DOSE_INPUTS[@]}"; do
    dose_cmd+=(--input "$input")
  done
  print_command "${dose_cmd[@]}"
  "${dose_cmd[@]}"
fi

mapfile -t TRANSFER_INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/transfer_from_*/layer_*/question_results.jsonl' -type f | sort)
if (( ${#TRANSFER_INPUTS[@]} > 0 )); then
  transfer_cmd=(
    "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
    --output-dir "$INTERVENTION_ROOT/transfer_aggregate"
    --n-bootstrap 2000
    --seed 5
  )
  for input in "${TRANSFER_INPUTS[@]}"; do
    transfer_cmd+=(--input "$input")
  done
  print_command "${transfer_cmd[@]}"
  "${transfer_cmd[@]}"
fi
