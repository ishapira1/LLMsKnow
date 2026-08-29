#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_hold_0828
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/confirmation/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/confirmation/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
[[ "${UNLOCK_CONFIRMATION:-0}" == 1 ]] || { printf 'Set UNLOCK_CONFIRMATION=1 only after freezing and reviewing the discovery specification.\n' >&2; exit 2; }
start_task_log confirmation_analysis all heldout_once none both
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT
command=("$PYTHON_BIN" "$RUNNER" analyze --output-root "$RESULT_ROOT" --scope confirmation --unlock-confirmation --bootstrap-replicates 10000 --permutation-replicates 10000)
print_command "${command[@]}"
"${command[@]}"
