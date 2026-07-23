#!/bin/bash
#SBATCH --job-name=weight_pruning_manifest
#SBATCH --partition=shared,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/manifests/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_pruning/paper_global_sharded_20260722/slurm/manifests/%x.%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
source "$REPO_DIR/jobs/sycophancy_pruning/paper_global_sharded_20260722/common.sh"
require_slurm_array_task
setup_environment

MODEL_INDEX="$SLURM_ARRAY_TASK_ID"
MODEL_ID="$(model_id_for_index "$MODEL_INDEX")"
MODEL_REVISION="$(model_revision_for_index "$MODEL_INDEX")"
require_sha "MODEL_REVISION" "$MODEL_REVISION"
if [[ -z "$ALPACA_DATA" ]]; then
  printf '%s\n' "ALPACA_DATA must be set to an explicit Alpaca JSON/JSONL path" >&2
  exit 1
fi
require_file "$ALPACA_DATA"
MODEL_SLUG="$(slugify "$MODEL_ID")"
BUNDLE_DIR="$(manifest_bundle_dir "$MODEL_ID" "$MODEL_REVISION")"

cmd=(
  "$ENV_PYTHON" scripts/build_strict_sycophancy_manifests.py
  --model-id "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --alpaca-data "$ALPACA_DATA"
  --output-dir "$BUNDLE_DIR"
)
for seed in 5 17 29; do
  for dataset in arc_challenge commonsense_qa; do
    run_dir="$(sampling_run_dir "$MODEL_ID" "$dataset" "$seed")"
    require_dir "$run_dir"
    cmd+=(--seed-records "$seed=$run_dir")
  done
done
for dataset in arc_challenge commonsense_qa; do
  evaluation_run_dir="$(sampling_run_dir "$MODEL_ID" "$dataset" 5)"
  cmd+=(--evaluation-records "$evaluation_run_dir")
done

TASK_LABEL="${MODEL_SLUG}_manifest"
STRUCTURED_DIR="$WEIGHT_PRUNING_LOG_ROOT/by_task/$MODEL_SLUG/manifests/job_${SLURM_JOB_ID}"
mkdir -p "$STRUCTURED_DIR"
exec > >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.out") \
  2> >(tee -a "$STRUCTURED_DIR/task_${MODEL_INDEX}.err" >&2)
printf -v command_text '%q ' "${cmd[@]}"
print_task_start "$TASK_LABEL" "$command_text"
printf '[manifests] model=%s revision=%s output=%s\n' "$MODEL_ID" "$MODEL_REVISION" "$BUNDLE_DIR"
START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
print_task_end "$TASK_LABEL" "$STATUS" "$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
