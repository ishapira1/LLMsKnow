#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --job-name=bonham_gpu_lanes_0918
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_multilane/%x_%j.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_multilane/%x_%j.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${STAGE:?STAGE is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"

LANES="${LANES:-${SLURM_NTASKS:-1}}"
GPUS_PER_LANE="${GPUS_PER_LANE:-1}"
CPUS_PER_LANE="${CPUS_PER_LANE:-4}"
MEM_PER_LANE="${MEM_PER_LANE:-48G}"
if ! [[ "$LANES" =~ ^[1-9][0-9]*$ && "$GPUS_PER_LANE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LANES and GPUS_PER_LANE must be positive integers\n' >&2
  exit 2
fi
case "$STAGE" in
  neutral_screen|n1_screen|source_screen)
    : "${PACK_START:?PACK_START is required}"
    : "${PACK_END:?PACK_END is required}"
    if ! [[ "$PACK_START" =~ ^[0-9]+$ && "$PACK_END" =~ ^[0-9]+$ ]] || (( PACK_START > PACK_END )); then
      printf 'PACK_START/PACK_END must be non-negative with start no greater than end\n' >&2
      exit 2
    fi
    ;;
  screen_sequence) ;;
  *) printf 'gpu_multilane supports screen stages only: %s\n' "$STAGE" >&2; exit 2 ;;
esac

PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
if (( GPUS_PER_LANE > 1 )); then
  export LLMSSYCOPH_DEVICE_MAP_AUTO=1
fi
cd "$REPO_DIR"
start_epoch="$(date +%s)"
printf 'experiment=pruning_bonham_20260918\nstage=%s\nmodel=%s\nlanes=%s\ngpus_per_lane=%s\nmem_per_lane=%s\npack_start=%s\npack_end=%s\n' \
  "$STAGE" "$MODEL_KEY" "$LANES" "$GPUS_PER_LANE" "$MEM_PER_LANE" "${PACK_START:-not_applicable}" "${PACK_END:-not_applicable}"
printf 'slurm_job_id=%s\nhostname=%s\nworking_directory=%s\npython=%s\nstart_time=%s\n' \
  "${SLURM_JOB_ID:-unset}" "$(hostname)" "$PWD" "$PYTHON_BIN" "$(date -Is)"
nvidia-smi || true

children=()
lane_logs="$LOG_ROOT/by_task/$MODEL_KEY/$STAGE/job_${SLURM_JOB_ID:-local}"
mkdir -p "$lane_logs" "$LOG_ROOT/slurm/gpu_multilane"
requeue_job() {
  printf 'received_usr1_requeue=1\n'
  for child in "${children[@]:-}"; do kill "$child" 2>/dev/null || true; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

for ((lane = 0; lane < LANES; lane++)); do
  if [[ "$STAGE" == screen_sequence ]]; then
    # This production mode intentionally covers both qualification banks.
    # Keep the stage list positional rather than comma-encoded in --export,
    # because Slurm uses commas to delimit exported variables.
    screen_stages=(n1_screen source_screen)
    command=(
      "$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" run-screen-sequence
      --result-root "$RESULT_ROOT"
      --model-key "$MODEL_KEY"
      --hf-cache "$HF_CACHE_DIR"
      --stages "${screen_stages[@]}"
      --lane-index "$lane"
      --lane-count "$LANES"
      --batch-size "${SCREEN_BATCH_SIZE:-4}"
    )
  else
    lane_start="$((PACK_START + lane))"
    if (( lane_start > PACK_END )); then
      continue
    fi
    command=(
      "$PYTHON_BIN" "$BUNDLE_DIR/campaign.py" run-screen-pack
      --result-root "$RESULT_ROOT"
      --model-key "$MODEL_KEY"
      --hf-cache "$HF_CACHE_DIR"
      --stage "$STAGE"
      --shard-start "$lane_start"
      --shard-end "$PACK_END"
      --shard-step "$LANES"
      --batch-size "${SCREEN_BATCH_SIZE:-4}"
    )
  fi
  printf 'lane=%s\n' "$lane"
  print_command "${command[@]}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_LANE" \
    --gpus-per-task="$GPUS_PER_LANE" --mem="$MEM_PER_LANE" \
    --output="$lane_logs/lane_${lane}.out" --error="$lane_logs/lane_${lane}.err" \
    "${command[@]}" &
  children+=("$!")
done

status=0
for child in "${children[@]}"; do
  if ! wait "$child"; then
    status=1
  fi
done
end_epoch="$(date +%s)"
printf 'end_time=%s\nexit_status=%s\nelapsed_seconds=%s\n' \
  "$(date -Is)" "$status" "$((end_epoch-start_epoch))"
sstat --jobs "$SLURM_JOB_ID" --format=JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInMax || true
exit "$status"
