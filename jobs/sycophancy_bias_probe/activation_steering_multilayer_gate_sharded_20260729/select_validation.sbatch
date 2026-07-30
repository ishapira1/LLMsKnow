#!/bin/bash
#SBATCH --job-name=syco_ml_select_20260729
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/selection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_multilayer_gate_sharded_20260729/slurm/selection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_multilayer_gate_sharded_20260729/common.sh
TASK_LABEL="both_models_arc_multilayer"
start_structured_task_log selection
LLAMA_RESULTS="$INTERVENTION_ROOT/models/llama31_8b/validation/question_results.jsonl"
QWEN_RESULTS="$INTERVENTION_ROOT/models/qwen25_7b/validation/question_results.jsonl"
test -s "$LLAMA_RESULTS"
test -s "$QWEN_RESULTS"
OUTPUT="$INTERVENTION_ROOT/validation_decision.json"
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
  select-multilayer-validation
  --input "$LLAMA_RESULTS"
  --input "$QWEN_RESULTS"
  --output "$OUTPUT"
  --n-bootstrap 2000
  --seed 5
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$OUTPUT"
