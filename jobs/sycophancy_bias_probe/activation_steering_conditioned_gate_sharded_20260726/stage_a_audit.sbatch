#!/bin/bash
#SBATCH --job-name=syco_cond_audit_20260726
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/audit/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/audit/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
start_structured_task_log audit

test ! -e "$AUDIT_OUTPUT_DIR" || {
  printf 'Refusing to overwrite Stage-A output: %s\n' "$AUDIT_OUTPUT_DIR" >&2
  exit 1
}
for input_path in \
  "$CONFIG_PATH" \
  "$QUESTION_MANIFEST" \
  "$LLAMA_DIRECTIONS" \
  "$QWEN_DIRECTIONS" \
  "$LLAMA_CSQA_SOURCE" \
  "$LLAMA_ARC_SOURCE" \
  "$QWEN_CSQA_SOURCE" \
  "$QWEN_ARC_SOURCE"; do
  test -e "$input_path" || {
    printf 'Missing Stage-A input: %s\n' "$input_path" >&2
    exit 1
  }
done

cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py" audit-mean-cancellation
  --config "$CONFIG_PATH"
  --question-manifest "$QUESTION_MANIFEST"
  --directions-path "$LLAMA_DIRECTIONS"
  --directions-path "$QWEN_DIRECTIONS"
  --source-run-dir "$LLAMA_CSQA_SOURCE"
  --source-run-dir "$LLAMA_ARC_SOURCE"
  --source-run-dir "$QWEN_CSQA_SOURCE"
  --source-run-dir "$QWEN_ARC_SOURCE"
  --output-dir "$AUDIT_OUTPUT_DIR"
  --n-folds 5
  --n-permutations 1000
  --n-bootstrap 2000
  --n-split-half 200
  --seed 5
)
print_command "${cmd[@]}"
"${cmd[@]}"
test -s "$AUDIT_OUTPUT_DIR/decision.json"
test -s "$AUDIT_OUTPUT_DIR/audit_report.md"
test -s "$AUDIT_OUTPUT_DIR/layer_table.csv"
