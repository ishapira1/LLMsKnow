#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_disc_0828
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/analysis/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/analysis/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
start_task_log discovery_analysis all audit_and_discovery none both
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT
audit=("$PYTHON_BIN" "$RUNNER" audit-coverage --output-root "$RESULT_ROOT")
analysis=("$PYTHON_BIN" "$RUNNER" analyze --output-root "$RESULT_ROOT" --scope discovery --bootstrap-replicates 10000 --permutation-replicates 10000)
print_command "${audit[@]}"
"${audit[@]}" || printf 'confirmation_coverage_status=incomplete\n'
print_command "${analysis[@]}"
"${analysis[@]}"
