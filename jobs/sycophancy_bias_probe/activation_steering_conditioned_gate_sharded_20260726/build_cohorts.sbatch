#!/bin/bash
#SBATCH --job-name=syco_cond_cohort_20260726
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/cohort/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/activation_steering_conditioned_gate_sharded_20260726/slurm/cohort/%x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail
source jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/common.sh
TASK_LABEL="both_models_arc"
start_structured_task_log cohort

CPU_DECISION="$AUDIT_OUTPUT_DIR/decision.json"
"$ENV_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["gpu_stage_authorized"] is True, "CPU gate did not authorize Stage B"' \
  "$CPU_DECISION"
mkdir -p "$INTERVENTION_ROOT/cohorts"
for model_index in 0 1; do
  resolve_conditioned_model "$model_index"
  test ! -e "$COHORT_MANIFEST" || {
    printf 'Refusing to overwrite cohort: %s\n' "$COHORT_MANIFEST" >&2
    exit 1
  }
  cmd=(
    "$ENV_PYTHON" "$REPO_DIR/run_activation_steering.py"
    build-conditioned-arc-cohort
    --source-run-dir "$ARC_SOURCE"
    --training-manifest "$QUESTION_MANIFEST"
    --output "$COHORT_MANIFEST"
    --maximum-per-split 120
  )
  print_command "${cmd[@]}"
  "${cmd[@]}"
  test -s "$COHORT_MANIFEST"
done
