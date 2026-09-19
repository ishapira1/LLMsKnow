#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --job-name=bonham_postmask_all_0918
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_postmask_all/%x_%j.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_postmask_all/%x_%j.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime

cd "$REPO_DIR"
start_epoch="$(date +%s)"
printf 'experiment=pruning_bonham_20260918\nstage=postmask_all_models\nslurm_job_id=%s\nhostname=%s\nworking_directory=%s\nstart_time=%s\n' \
  "${SLURM_JOB_ID:-unset}" "$(hostname)" "$PWD" "$(date -Is)"
nvidia-smi || true

children=()
lane_logs="$LOG_ROOT/by_task/all_models/postmask_all/job_${SLURM_JOB_ID:-local}"
mkdir -p "$lane_logs" "$LOG_ROOT/slurm/gpu_postmask_all"
requeue_job() {
  printf 'received_usr1_requeue=1\n'
  for child in "${children[@]:-}"; do kill "$child" 2>/dev/null || true; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

models=(qwen25_7b llama31_8b gemma4_12b)
gpu_counts=(1 1 2)
memory=(48G 48G 96G)
for index in 0 1 2; do
  model="${models[$index]}"
  gpus="${gpu_counts[$index]}"
  printf 'lane=%s model=%s gpus=%s memory=%s\n' \
    "$index" "$model" "$gpus" "${memory[$index]}"
  device_map_auto=0
  if (( gpus > 1 )); then device_map_auto=1; fi
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gpus-per-task="$gpus" --mem="${memory[$index]}" \
    --output="$lane_logs/${model}.out" --error="$lane_logs/${model}.err" \
    env MODEL_KEY="$model" LLMSSYCOPH_DEVICE_MAP_AUTO="$device_map_auto" \
    "$BUNDLE_DIR/gpu_postmask_lane.sh" &
  children+=("$!")
done

status=0
for child in "${children[@]}"; do
  if ! wait "$child"; then status=1; fi
done
end_epoch="$(date +%s)"
printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
  "$(date -Is)" "$status" "$((end_epoch-start_epoch))"
sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
exit "$status"
