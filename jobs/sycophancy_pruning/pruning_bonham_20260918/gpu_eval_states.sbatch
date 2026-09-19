#!/usr/bin/env bash
#SBATCH --account=barak_lab
#SBATCH --job-name=bonham_eval_states_0918
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu
#SBATCH --output=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_eval_states/%x_%j.out
#SBATCH --error=/n/home12/ishapira/LLMsKnow/jobs/sycophancy_bias_probe/logs/pruning_bonham_20260918/slurm/gpu_eval_states/%x_%j.err

set -Eeuo pipefail
: "${BONHAM_BUNDLE_DIR:?BONHAM_BUNDLE_DIR is required}"
source "$BONHAM_BUNDLE_DIR/common.sh"
require_runtime
: "${MODEL_KEY:?MODEL_KEY is required}"

STATE_OFFSET="${STATE_OFFSET:-0}"
STATE_COUNT="${STATE_COUNT:-${SLURM_NTASKS:-1}}"
STATE_INDICES="${STATE_INDICES:-}"
GPUS_PER_STATE="${GPUS_PER_STATE:-1}"
CPUS_PER_STATE="${CPUS_PER_STATE:-4}"
MEM_PER_STATE="${MEM_PER_STATE:-48G}"
if ! [[ "$STATE_OFFSET" =~ ^[0-9]+$ && "$STATE_COUNT" =~ ^[1-9][0-9]*$ && "$GPUS_PER_STATE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STATE_OFFSET must be non-negative; STATE_COUNT/GPUS_PER_STATE must be positive\n' >&2
  exit 2
fi
state_indices=()
if [[ -n "$STATE_INDICES" ]]; then
  IFS=':' read -r -a state_indices <<< "$STATE_INDICES"
  declare -A seen_state_indices=()
  for state_index in "${state_indices[@]}"; do
    if ! [[ "$state_index" =~ ^[0-7]$ ]] || [[ -n "${seen_state_indices[$state_index]:-}" ]]; then
      printf 'STATE_INDICES must contain unique colon-separated indices from 0 through 7: %s\n' \
        "$STATE_INDICES" >&2
      exit 2
    fi
    seen_state_indices[$state_index]=1
  done
  STATE_COUNT="${#state_indices[@]}"
else
  if (( STATE_OFFSET + STATE_COUNT > 8 )); then
    printf 'Requested state range exceeds the eight frozen states\n' >&2
    exit 2
  fi
  for ((within = 0; within < STATE_COUNT; within++)); do
    state_indices+=("$((STATE_OFFSET + within))")
  done
fi
if (( GPUS_PER_STATE > 1 )); then
  export LLMSSYCOPH_DEVICE_MAP_AUTO=1
fi
PYTHON_BIN="$(python_for_model "$MODEL_KEY")"; export PYTHON_BIN
# Keep the preset scalar-valued: Slurm uses commas to delimit --export entries.
# The default preserves the full original runner.  The paper-core preset lets the
# accelerated campaign finish the Section 4/5 analyses before the capability tail.
case "${EVALUATION_FAMILY_SET:-all}" in
  all) evaluation_families=(generalization useful_assertions source_attribution capabilities) ;;
  paper_core) evaluation_families=(generalization useful_assertions source_attribution) ;;
  source_attribution) evaluation_families=(source_attribution) ;;
  capabilities) evaluation_families=(capabilities) ;;
  *)
    printf 'EVALUATION_FAMILY_SET must be all, paper_core, source_attribution, or capabilities: %s\n' \
      "$EVALUATION_FAMILY_SET" >&2
    exit 2
    ;;
esac
states=(
  unpruned
  n1_mechanism
  n2_selective
  random_n1
  random_n2
  weak_prompt
  strong_prompt
  prompt_only_meandiff
)

cd "$REPO_DIR"
start_epoch="$(date +%s)"
printf 'experiment=pruning_bonham_20260918\nstage=evaluation_state_sequence\nmodel=%s\nstate_offset=%s\nstate_indices=%s\nstate_count=%s\ngpus_per_state=%s\nmem_per_state=%s\nfamilies=%s\n' \
  "$MODEL_KEY" "$STATE_OFFSET" "${state_indices[*]}" "$STATE_COUNT" "$GPUS_PER_STATE" "$MEM_PER_STATE" "${evaluation_families[*]}"
printf 'slurm_job_id=%s\nhostname=%s\nworking_directory=%s\npython=%s\nstart_time=%s\n' \
  "${SLURM_JOB_ID:-unset}" "$(hostname)" "$PWD" "$PYTHON_BIN" "$(date -Is)"
nvidia-smi || true

children=()
state_logs="$LOG_ROOT/by_task/$MODEL_KEY/evaluation_state_sequence/job_${SLURM_JOB_ID:-local}"
mkdir -p "$state_logs" "$LOG_ROOT/slurm/gpu_eval_states"
requeue_job() {
  printf 'received_usr1_requeue=1\n'
  for child in "${children[@]:-}"; do kill "$child" 2>/dev/null || true; done
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

for state_index in "${state_indices[@]}"; do
  state_id="${states[$state_index]}"
  command=(
    "$PYTHON_BIN" "$BUNDLE_DIR/evaluations.py" run-state-sequence
    --result-root "$RESULT_ROOT"
    --model-key "$MODEL_KEY"
    --state-id "$state_id"
    --families "${evaluation_families[@]}"
    --hf-cache "$HF_CACHE_DIR"
    --batch-size "${EVALUATION_BATCH_SIZE:-4}"
  )
  printf 'state_index=%s\nstate_id=%s\n' "$state_index" "$state_id"
  print_command "${command[@]}"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_STATE" \
    --gpus-per-task="$GPUS_PER_STATE" --mem="$MEM_PER_STATE" \
    --output="$state_logs/state_${state_index}_${state_id}.out" \
    --error="$state_logs/state_${state_index}_${state_id}.err" \
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
