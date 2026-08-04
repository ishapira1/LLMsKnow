#!/usr/bin/env bash
#SBATCH --job-name=syco_dense_cpu_0803
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -Eeuo pipefail
source "${DENSE_BUNDLE_DIR:?DENSE_BUNDLE_DIR required}/common.sh"
STAGE="${DENSE_STAGE:?DENSE_STAGE required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
cd "$REPO_DIR"
start_task_log "$STAGE" "$STAGE"
on_exit() {
  status=$?
  finish_task_log "$status"
  if [[ "$status" -ne 0 ]]; then
    { printf 'Experiment: dense_behavioral_diversity_20260803\nStage: %s\nExit code: %s\nLog: %s\n' "$STAGE" "$status" "$TASK_LOG_PATH"; } |
      /usr/bin/mail -s "[dense_behavioral_diversity] $STAGE | FAILED" "$EMAIL_TO" || true
  fi
  exit "$status"
}
trap on_exit EXIT

case "$STAGE" in
  preflight)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" preflight --model-snapshot "$MODEL_SNAPSHOT" --original-prune "$ORIGINAL_PRUNE" --original-preserve "$ORIGINAL_PRESERVE" --previous-inputs "$SIZE_ROOT/inputs" --broad-preserve "$BROAD_PRESERVE" --sampling-records "$SAMPLING_ARC" "$SAMPLING_CSQA" --output "$RESULT_ROOT/preflight.json")
    ;;
  build_candidates)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" build-candidates --sampling-records "$SAMPLING_ARC" "$SAMPLING_CSQA" --original-prune "$ORIGINAL_PRUNE" --original-preserve "$ORIGINAL_PRESERVE" --output-root "$RESULT_ROOT/candidates")
    ;;
  build_manifests)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" build-manifests --sampled-shards "$RESULT_ROOT/sampling/shard_0.jsonl" "$RESULT_ROOT/sampling/shard_1.jsonl" "$RESULT_ROOT/sampling/shard_2.jsonl" "$RESULT_ROOT/sampling/shard_3.jsonl" --candidate-questions "$RESULT_ROOT/candidates/candidate_questions.jsonl" --sampling-records "$SAMPLING_ARC" "$SAMPLING_CSQA" --original-prune "$ORIGINAL_PRUNE" --original-preserve "$ORIGINAL_PRESERVE" --broad-preserve "$BROAD_PRESERVE" --previous-inputs "$SIZE_ROOT/inputs" --output-root "$INPUTS_ROOT")
    ;;
  email_manifests)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/email_status.py" --stage manifests --result-root "$RESULT_ROOT" --email-to "$EMAIL_TO")
    ;;
  build_masks)
    unit="${UNITS[$TASK_ID]}"
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" build-masks --matrix "$INPUTS_ROOT/$unit/config_matrix.json" --result-root "$RESULT_ROOT")
    ;;
  replicate_behavior)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" replicate-base --kind behavior --result-root "$RESULT_ROOT")
    ;;
  replicate_utility)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" replicate-base --kind utility --result-root "$RESULT_ROOT")
    ;;
  select)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" select-finalists --inputs-root "$INPUTS_ROOT" --result-root "$RESULT_ROOT" --output "$RESULT_ROOT/selection/finalists.json")
    ;;
  email_screening)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/email_status.py" --stage screening --result-root "$RESULT_ROOT" --email-to "$EMAIL_TO")
    ;;
  build_states)
    command=("$PYTHON_BIN" "$CAMPAIGN_PY" build-states --selection "$RESULT_ROOT/selection/finalists.json" --previous-states "$DIVERSE_ROOT/states.json" --result-root "$RESULT_ROOT" --output "$STATES_CONFIG")
    ;;
  score_elephant)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/score_elephant.py" --states-config "$STATES_CONFIG" --generation-root "$RESULT_ROOT/nonfactual/generation/elephant" --output "$RESULT_ROOT/nonfactual/elephant_summary.json")
    ;;
  aggregate)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/aggregate_results.py" --result-root "$RESULT_ROOT" --states-config "$STATES_CONFIG")
    ;;
  email_final)
    command=("$PYTHON_BIN" "$BUNDLE_DIR/email_status.py" --stage final --result-root "$RESULT_ROOT" --email-to "$EMAIL_TO")
    ;;
  *) printf 'Unknown CPU stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
print_command "${command[@]}"
"${command[@]}"
