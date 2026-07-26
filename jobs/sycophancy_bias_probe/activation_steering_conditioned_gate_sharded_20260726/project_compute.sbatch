#!/bin/bash
#SBATCH --job-name=syco_cond_project_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/projection/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/projection/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_LABEL="both_models_arc"
start_structured_task_log compute_projection

OUTPUT="$INTERVENTION_ROOT/compute_projection.json"
test ! -e "$OUTPUT" || {
  printf 'Refusing to overwrite compute projection: %s\n' "$OUTPUT" >&2
  exit 1
}
cmd=(
  "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
  project-conditioned-compute
  --benchmark-manifest "$INTERVENTION_ROOT/stage_b/llama31_8b/bf16_gate/manifest.json"
  --benchmark-manifest "$INTERVENTION_ROOT/stage_b/qwen25_7b/bf16_gate/manifest.json"
  --validation-questions-per-model 120
  --test-questions-per-model 120
  --output "$OUTPUT"
)
print_command "${cmd[@]}"
"${cmd[@]}"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["gpu_submission_authorized"] is True, d' \
  "$OUTPUT"
