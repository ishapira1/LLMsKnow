#!/bin/bash
#SBATCH --job-name=syco_sig_agg_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/aggregate/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_signal_sharded_20260726/slurm/aggregate/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_signal_sharded_20260726/common.sh
TASK_LABEL="signal_final"
start_structured_task_log signal_aggregate

mapfile -t TEST_INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/test_selected/layer_*/question_results.jsonl' -type f | sort)
mapfile -t PROBE_INPUTS < <(find "$INTERVENTION_ROOT/models" -path '*/fixed_probe_test/layer_*/question_results.jsonl' -type f | sort)
(( ${#TEST_INPUTS[@]} >= 8 )) || {
  printf 'Expected selected-layer test outputs, found %s\n' "${#TEST_INPUTS[@]}" >&2
  exit 1
}
(( ${#PROBE_INPUTS[@]} == 2 )) || {
  printf 'Expected two CSQA fixed-probe outputs, found %s\n' "${#PROBE_INPUTS[@]}" >&2
  exit 1
}
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" aggregate
  --output-dir "$INTERVENTION_ROOT/final_aggregate"
  --n-bootstrap 1000
  --seed 26
)
for input in "${TEST_INPUTS[@]}" "${PROBE_INPUTS[@]}"; do
  cmd+=(--input "$input")
done
print_command "${cmd[@]}"
"${cmd[@]}"

summary_cmd=(
  "$ENV_PYTHON" "$REPO_DIR/scripts/summarize_activation_steering_signal.py"
  --selection "$INTERVENTION_ROOT/dev_aggregate/layer_selection.json"
  --output "$INTERVENTION_ROOT/final_aggregate/signal_summary.json"
)
for input in "${TEST_INPUTS[@]}" "${PROBE_INPUTS[@]}"; do
  summary_cmd+=(--input "$input")
done
print_command "${summary_cmd[@]}"
"${summary_cmd[@]}"
test -s "$INTERVENTION_ROOT/final_aggregate/signal_summary.json"
test -s "$INTERVENTION_ROOT/final_aggregate/signal_summary.md"
