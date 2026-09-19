#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --job-name=bonham_gpu_pack_0918
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_packed/%x_%j.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_packed/%x_%j.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
: "${STAGE:?STAGE is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
: "${PACK_START:?PACK_START is required}"
: "${PACK_END:?PACK_END is required}"

if ! [[ "$PACK_START" =~ ^[0-9]+$ && "$PACK_END" =~ ^[0-9]+$ ]]; then
  printf 'PACK_START and PACK_END must be non-negative integers\n' >&2
  exit 2
fi
if (( PACK_START > PACK_END )); then
  printf 'PACK_START must be no greater than PACK_END\n' >&2
  exit 2
fi

printf 'packed_stage=%s\nmodel=%s\npack_start=%s\npack_end=%s\nslurm_job_id=%s\nhostname=%s\nstart_time=%s\n' \
  "$STAGE" "$MODEL_KEY" "$PACK_START" "$PACK_END" "${SLURM_JOB_ID:-unknown}" \
  "$(hostname -f)" "$(date --iso-8601=seconds)"
nvidia-smi || true

trap 'printf "received_usr1_requeue=1\n"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

for ((task = PACK_START; task <= PACK_END; task++)); do
  printf 'packed_task_start=%s time=%s\n' "$task" "$(date --iso-8601=seconds)"
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID:-0}" \
  SLURM_ARRAY_TASK_ID="$task" \
    bash "$BONHAM_BUNDLE_DIR/gpu_array.sbatch"
  printf 'packed_task_end=%s time=%s\n' "$task" "$(date --iso-8601=seconds)"
done

printf 'packed_complete=1\nend_time=%s\n' "$(date --iso-8601=seconds)"
