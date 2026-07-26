#!/bin/bash
#SBATCH --job-name=syco_cond_select_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/selection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/selection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_LABEL="both_models_arc"
start_structured_task_log validation_selection

OUTPUT="$INTERVENTION_ROOT/validation_selection.json"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
  select-conditioned-validation
  --input "$INTERVENTION_ROOT/stage_b/llama31_8b/validation/question_results.jsonl"
  --input "$INTERVENTION_ROOT/stage_b/qwen25_7b/validation/question_results.jsonl"
  --cpu-decision "$AUDIT_OUTPUT_DIR/decision.json"
  --output "$OUTPUT"
  --n-bootstrap 2000
  --seed 5
)
print_command "${cmd[@]}"
"${cmd[@]}"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert any(x["selected"] for x in d["selections"]), "No model has an eligible validation candidate"' \
  "$OUTPUT"
