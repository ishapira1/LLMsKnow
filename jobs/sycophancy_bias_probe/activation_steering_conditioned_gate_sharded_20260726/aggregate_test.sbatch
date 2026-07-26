#!/bin/bash
#SBATCH --job-name=syco_cond_agg_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/aggregate/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/aggregate/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_LABEL="both_models_arc"
start_structured_task_log heldout_aggregate

SELECTED_MODELS="$(
  "$ENV_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(sum(bool(x["selected"]) for x in d["selections"]))' \
    "$INTERVENTION_ROOT/validation_selection.json"
)"
(( SELECTED_MODELS >= 1 )) || {
  printf 'No model has an eligible validation candidate\n' >&2
  exit 1
}
mapfile -t INPUTS < <(
  find "$INTERVENTION_ROOT/stage_b" \
    \( -path '*/test_learned/layer_*/question_results.jsonl' \
    -o -path '*/test_controls/seed_*/question_results.jsonl' \
    -o -path '*/suffix_same_per_position/question_results.jsonl' \) \
    -type f | sort
)
EXPECTED_INPUTS=$((SELECTED_MODELS * 23))
(( ${#INPUTS[@]} == EXPECTED_INPUTS )) || {
  printf 'Expected %s held-out inputs (23 per selected model), found %s\n' \
    "$EXPECTED_INPUTS" "${#INPUTS[@]}" >&2
  exit 1
}
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
  aggregate-conditioned-test
  --selection "$INTERVENTION_ROOT/validation_selection.json"
  --output-dir "$INTERVENTION_ROOT/final_aggregate"
  --n-bootstrap 2000
  --seed 5
)
for input_path in "${INPUTS[@]}"; do
  cmd+=(--input "$input_path")
done
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$INTERVENTION_ROOT/final_aggregate/final_decision.json"
