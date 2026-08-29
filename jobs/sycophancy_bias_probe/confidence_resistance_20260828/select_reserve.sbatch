#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_rsel_0828
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/reserve_select/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/reserve_select/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
start_task_log reserve_select all common_qc_prefix none both
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT
selection=("$PYTHON_BIN" "$RUNNER" select-reserve --output-root "$RESULT_ROOT")
audit=("$PYTHON_BIN" "$RUNNER" audit-coverage --output-root "$RESULT_ROOT")
print_command "${selection[@]}"
"${selection[@]}"
print_command "${audit[@]}"
"${audit[@]}"
