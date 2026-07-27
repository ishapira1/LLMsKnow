#!/bin/bash
#SBATCH --job-name=syco_cond_stop_20260727
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/aggregate/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/aggregate/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_LABEL="both_models_arc_validation_stop"
start_structured_task_log validation_stop_finalization

OUTPUT_DIR="$INTERVENTION_ROOT/final_aggregate"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
  finalize-conditioned-validation-stop
  --input "$INTERVENTION_ROOT/stage_b/llama31_8b/validation/question_results.jsonl"
  --input "$INTERVENTION_ROOT/stage_b/qwen25_7b/validation/question_results.jsonl"
  --selection "$INTERVENTION_ROOT/validation_selection.json"
  --cpu-audit-dir "$AUDIT_OUTPUT_DIR"
  --output-dir "$OUTPUT_DIR"
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT_DIR/final_decision.json"
test -s "$OUTPUT_DIR/final_report.md"
test -s "$OUTPUT_DIR/model_results.csv"
test -s "$OUTPUT_DIR/validation_candidates.csv"
test -s "$OUTPUT_DIR/validation_gate.png"
