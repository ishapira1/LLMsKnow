#!/usr/bin/env bash
#SBATCH --job-name=syco_dense_gpu_0803
#SBATCH --partition=gpu_h200,seas_gpu,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -Eeuo pipefail
source "${DENSE_BUNDLE_DIR:?DENSE_BUNDLE_DIR required}/common.sh"
STAGE="${DENSE_STAGE:?DENSE_STAGE required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
cd "$REPO_DIR"
label="$STAGE"
case "$STAGE" in
  sample)
    label="sampling/shard_$TASK_ID"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" sample-candidates --candidates "$RESULT_ROOT/candidates/candidate_prompts_shard_$TASK_ID.jsonl" --model-snapshot "$MODEL_SNAPSHOT" --batch-size 16 --output "$RESULT_ROOT/sampling/shard_$TASK_ID.jsonl")
    ;;
  score)
    if [[ "$TASK_ID" -lt 4 ]]; then role="prune"; shard="$TASK_ID"; else role="preserve"; shard="$((TASK_ID-4))"; fi
    label="scores/components/$role/shard_$shard"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" family-score-shard --role "$role" --shard-index "$shard" --manifest "$INPUTS_ROOT/components/${role}_all.jsonl" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT")
    ;;
  tune_base)
    unit="${UNITS[0]}"; label="tune/base"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" evaluate --unit "$unit" --cohort tune --base --matrix "$INPUTS_ROOT/$unit/config_matrix.json" --inputs-root "$INPUTS_ROOT/$unit" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --hf-cache-dir "$HF_CACHE_DIR")
    ;;
  tune_grid)
    unit_index="$((TASK_ID / 9))"; config_index="$((TASK_ID % 9))"; unit="${UNITS[$unit_index]}"; matrix="$INPUTS_ROOT/$unit/config_matrix.json"
    config_id="$("$PYTHON_BIN" "$CAMPAIGN_PY" config-field --matrix "$matrix" --config-index "$config_index" --field config_id)"; label="tune/$unit/$config_id"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" evaluate --unit "$unit" --cohort tune --config-index "$config_index" --matrix "$matrix" --inputs-root "$INPUTS_ROOT/$unit" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --hf-cache-dir "$HF_CACHE_DIR")
    ;;
  tune_matched)
    unit="${UNITS[$TASK_ID]}"; matrix="$INPUTS_ROOT/$unit/matched_matrix.json"; label="tune/$unit/matched996"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" evaluate --unit "$unit" --cohort tune --config-index 0 --matrix "$matrix" --inputs-root "$INPUTS_ROOT/$unit" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --hf-cache-dir "$HF_CACHE_DIR")
    ;;
  utility_base)
    unit="${UNITS[0]}"; label="utility/base"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" quick-utility --unit "$unit" --base --matrix "$INPUTS_ROOT/$unit/config_matrix.json" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --alpaca-manifest "$INPUTS_ROOT/alpaca_heldout_512.jsonl")
    ;;
  utility_grid)
    unit_index="$((TASK_ID / 9))"; config_index="$((TASK_ID % 9))"; unit="${UNITS[$unit_index]}"; matrix="$INPUTS_ROOT/$unit/config_matrix.json"; label="utility/$unit/$config_index"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" quick-utility --unit "$unit" --config-index "$config_index" --matrix "$matrix" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --alpaca-manifest "$INPUTS_ROOT/alpaca_heldout_512.jsonl")
    ;;
  utility_matched)
    unit="${UNITS[$TASK_ID]}"; matrix="$INPUTS_ROOT/$unit/matched_matrix.json"; label="utility/$unit/matched996"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" quick-utility --unit "$unit" --config-index 0 --matrix "$matrix" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --alpaca-manifest "$INPUTS_ROOT/alpaca_heldout_512.jsonl")
    ;;
  final_suite)
    case "$TASK_ID" in
      0) label="full/factual"; command=("$PYTHON_BIN" "$DIVERSE_BUNDLE/run_previous.py" "$PREVIOUS_BUNDLE/multi_factual.py" --manifest "$INPUTS_ROOT/final_manifest.jsonl" --expected-questions 1133 --result-name factual --states-config "$STATES_CONFIG" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT") ;;
      1) label="full/factual_paraphrase"; command=("$PYTHON_BIN" "$DIVERSE_BUNDLE/run_previous.py" "$PREVIOUS_BUNDLE/multi_factual.py" --manifest "$INPUTS_ROOT/final_paraphrase_manifest.jsonl" --expected-questions 694 --result-name factual_paraphrase --states-config "$STATES_CONFIG" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT") ;;
      2) label="full/utility"; command=("$PYTHON_BIN" "$DIVERSE_BUNDLE/run_previous.py" "$PREVIOUS_BUNDLE/multi_utility.py" --states-config "$STATES_CONFIG" --result-root "$RESULT_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --alpaca-manifest "$INPUTS_ROOT/alpaca_heldout_512.jsonl") ;;
      3|4|5)
        tasks=(mmlu icl sycobench); task="${tasks[$((TASK_ID-3))]}"; label="full/$task"
        command=("$PYTHON_BIN" "$DIVERSE_BUNDLE/run_previous.py" "$PREVIOUS_BUNDLE/multi_capability.py" --task "$task" --capability-audit "$CAPABILITY_AUDIT" --inputs-root "$CAPABILITY_INPUTS" --result-root "$RESULT_ROOT" --states-config "$STATES_CONFIG" --model-snapshot "$MODEL_SNAPSHOT" --sycobench-source "$SYCOBENCH_SOURCE" --email-to "$EMAIL_TO")
        ;;
      6) label="full/elephant"; command=("$PYTHON_BIN" "$DIVERSE_BUNDLE/run_previous.py" "$PREVIOUS_BUNDLE/multi_nonfactual.py" --experiment-py "$NONFACTUAL_EXPERIMENT_PY" --states-config "$STATES_CONFIG" generate --benchmark elephant --inputs-root "$NONFACTUAL_INPUTS" --result-root "$RESULT_ROOT/nonfactual" --model-snapshot "$MODEL_SNAPSHOT" --attempt-id initial) ;;
      *) printf 'Unknown final suite task: %s\n' "$TASK_ID" >&2; exit 2 ;;
    esac
    ;;
  *) printf 'Unknown GPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
start_task_log "$STAGE" "$label"
on_exit() {
  status=$?
  finish_task_log "$status"
  if [[ "$status" -ne 0 ]]; then
    { printf 'Experiment: dense_behavioral_diversity_20260803\nStage: %s\nTask: %s\nExit code: %s\nLog: %s\n' "$STAGE" "$label" "$status" "$TASK_LOG_PATH"; } |
      /usr/bin/mail -s "[dense_behavioral_diversity] $label | FAILED" "$EMAIL_TO" || true
  fi
  exit "$status"
}
trap on_exit EXIT
print_command "${command[@]}"
"${command[@]}"
