#!/usr/bin/env bash
#SBATCH --job-name=syco_conf_gpt_0828
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/gpt/%x.%j.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/confidence_resistance_20260828/slurm/gpt/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
set -Eeuo pipefail
source "${CONFIDENCE_RESISTANCE_BUNDLE_DIR:?required}/common.sh"
require_runner
[[ "${OPENAI_CONFIRM_SPEND:-0}" == 1 ]] || { printf 'OPENAI_CONFIRM_SPEND=1 is required\n' >&2; exit 2; }
[[ "${PRICING_RECHECKED:-0}" == 1 ]] || { printf 'Recheck official OpenAI Batch pricing and set PRICING_RECHECKED=1\n' >&2; exit 2; }
start_task_log gpt all_gpt pilot_neutral_endorsed gpt both
trap 'status=$?; finish_task_log "$status"; exit "$status"' EXIT

for stage in pilot-biased pilot-unbiased; do
  command=("$PYTHON_BIN" "$RUNNER" run-gpt-batch --output-root "$RESULT_ROOT" --gpt-stage "$stage" --confirm-spend)
  print_command "${command[@]}"
  "${command[@]}"
done
audit=("$PYTHON_BIN" "$RUNNER" audit-gpt-pilot --output-root "$RESULT_ROOT")
print_command "${audit[@]}"
"${audit[@]}"
pilot_passed="$("$PYTHON_BIN" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["passed"])))' "$RESULT_ROOT/audit/gpt_pilot_audit.json")"
fallback=()
if [[ "$pilot_passed" != 1 ]]; then
  fallback=(--gpt-unbiased)
  printf 'gpt_scoring_mode=unbiased_top20_censored_fallback\n'
else
  printf 'gpt_scoring_mode=equal_bias_point_identified\n'
fi
for dataset in arc_challenge commonsense_qa; do
  command=("$PYTHON_BIN" "$RUNNER" run-gpt-batch --output-root "$RESULT_ROOT" --gpt-stage neutral --dataset "$dataset" --confirm-spend "${fallback[@]}")
  print_command "${command[@]}"
  "${command[@]}"
done
freeze=("$PYTHON_BIN" "$RUNNER" freeze-targets --output-root "$RESULT_ROOT" --model-key gpt)
print_command "${freeze[@]}"
"${freeze[@]}"
for dataset in arc_challenge commonsense_qa; do
  command=("$PYTHON_BIN" "$RUNNER" run-gpt-batch --output-root "$RESULT_ROOT" --gpt-stage endorsed --dataset "$dataset" --confirm-spend "${fallback[@]}")
  print_command "${command[@]}"
  "${command[@]}"
done
