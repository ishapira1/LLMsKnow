#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --partition=gpu_test
#SBATCH --job-name=bonham_model_pipe_0918
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:4
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_model_pipeline/%x_%j.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_model_pipeline/%x_%j.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${MODEL_KEY:?MODEL_KEY is required}"

states=(
  unpruned n1_mechanism n2_selective random_n1
  random_n2 weak_prompt strong_prompt prompt_only_meandiff
)
lane_root="$LOG_ROOT/by_task/$MODEL_KEY/gpu_model_pipeline/job_${SLURM_JOB_ID:-local}"
mkdir -p "$lane_root" "$LOG_ROOT/slurm/gpu_model_pipeline"
cd "$REPO_DIR"
start_epoch="$(date +%s)"
printf 'experiment=pruning_bonham_20260918\nstage=gpu_model_pipeline\nmodel=%s\nstart_time=%s\nhostname=%s\n' \
  "$MODEL_KEY" "$(date -Is)" "$(hostname)"
nvidia-smi || true

children=()
evaluation_pids=()
LAST_PID=''
launch_step() {
  local label="$1"
  shift
  printf 'launch_step=%s command=' "$label"
  printf '%q ' "$@"
  printf '\n'
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gpus-per-task=1 --mem=48G \
    --output="$lane_root/${label}.out" --error="$lane_root/${label}.err" \
    "$BUNDLE_DIR/gpu_model_pipeline_lane.sh" "$@" &
  LAST_PID="$!"
  children+=("$LAST_PID")
}

requeue_job() {
  printf 'received_usr1_requeue=1\n'
  for child in "${children[@]:-}"; do kill "$child" 2>/dev/null || true; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

# Fill all four slices immediately.  The two initial evaluation states were
# written by model smoke and do not depend on random controls or MeanDiff; the
# mask states are queued as soon as their prerequisite lane completes.
launch_step random_and_states random_and_states
random_pid="$LAST_PID"
launch_step steering steering
steering_pid="$LAST_PID"
launch_step paper_unpruned evaluation unpruned paper_core
evaluation_pids+=("$LAST_PID")
launch_step paper_weak_prompt evaluation weak_prompt paper_core
evaluation_pids+=("$LAST_PID")

if ! wait "$random_pid"; then
  printf 'pipeline_prerequisite_failed=random_and_states\n' >&2
  exit 1
fi
for state_id in n1_mechanism n2_selective random_n1 random_n2 strong_prompt; do
  launch_step "paper_${state_id}" evaluation "$state_id" paper_core
  evaluation_pids+=("$LAST_PID")
done

if ! wait "$steering_pid"; then
  printf 'pipeline_prerequisite_failed=steering\n' >&2
  exit 1
fi
launch_step paper_prompt_only_meandiff evaluation prompt_only_meandiff paper_core
evaluation_pids+=("$LAST_PID")

status=0
for child in "${evaluation_pids[@]}"; do
  if ! wait "$child"; then status=1; fi
done
if (( status != 0 )); then
  printf 'pipeline_stage_failed=paper_core\n' >&2
  exit "$status"
fi
printf 'paper_core_complete=1 model=%s time=%s\n' "$MODEL_KEY" "$(date -Is)"

# Capability jobs use the same four-slice allocation in two automatically
# scheduled waves, avoiding another queue handoff.
capability_pids=()
for state_id in "${states[@]}"; do
  launch_step "capabilities_${state_id}" evaluation "$state_id" capabilities
  capability_pids+=("$LAST_PID")
done
status=0
for child in "${capability_pids[@]}"; do
  if ! wait "$child"; then status=1; fi
done
end_epoch="$(date +%s)"
printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
  "$(date -Is)" "$status" "$((end_epoch-start_epoch))"
sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
exit "$status"
