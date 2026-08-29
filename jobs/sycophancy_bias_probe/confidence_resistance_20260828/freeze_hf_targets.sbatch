#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_tgt_0828
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/freeze_targets/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/freeze_targets/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
start_task_log freeze_hf_targets local freeze_model_specific_ranks none both
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT
for model_key in llama qwen; do
  command=("$PYTHON_BIN" "$RUNNER" freeze-targets --output-root "$RESULT_ROOT" --model-key "$model_key")
  print_command "${command[@]}"
  "${command[@]}"
done
