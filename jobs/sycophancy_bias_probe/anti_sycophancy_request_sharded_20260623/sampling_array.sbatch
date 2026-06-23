#!/bin/bash
#SBATCH --job-name=syco_antisyc_samp_20260623
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/sampling/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/sampling/%x.%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

BUNDLE_NAME="anti_sycophancy_request_sharded_20260623"
SCRIPT_TAG="anti-sycophancy-sampling-20260623"
REQUEST_TAG="${REQUEST_TAG:-20260623}"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load python/3.10.9-fasrc01

ENV_PYTHON="${ENV_PYTHON:-/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python}"
if [[ ! -x "$ENV_PYTHON" ]]; then
  printf '%s\n' "Missing python interpreter: $ENV_PYTHON" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  printf '%s\n' "Missing .env in $REPO_DIR" >&2
  exit 1
fi
set -a
source .env
set +a

if [[ -n "${SYCOPHANCY_STORAGE_ROOT_OVERRIDE:-}" ]]; then
  export SYCOPHANCY_STORAGE_ROOT="$SYCOPHANCY_STORAGE_ROOT_OVERRIDE"
  unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HOME
  unset TRITON_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
  unset TMPDIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME OUT_DIR LOG_ROOT
fi

source jobs/sycophancy_bias_probe/storage_common.sh
configure_sycophancy_bias_storage "$BUNDLE_NAME"
HF_CACHE_DIR="$HF_HUB_CACHE"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  "$TRITON_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$TMPDIR" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$OUT_DIR" \
  "$LOG_ROOT" \
  "$LOG_ROOT/slurm/sampling" \
  "$LOG_ROOT/by_task"

DEFAULT_BIAS_TYPES_CSV="$("$ENV_PYTHON" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=False)))')"
BIAS_TYPES_CSV="${BIAS_TYPES_CSV:-$DEFAULT_BIAS_TYPES_CSV}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
SPLIT_SEED="${SPLIT_SEED:-5}"
SEED="${SEED:-5}"
PROBE_SEED="${PROBE_SEED:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"
ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}"
TASK_FILTER="${TASK_FILTER:-}"
REQUEST_FILTER="${REQUEST_FILTER:-}"

if [[ -z "$BIAS_TYPES_CSV" ]]; then
  printf '%s\n' "No trainable non-neutral prompt families resolved." >&2
  exit 1
fi

TASK_LABELS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)
MODEL_IDS=(
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen2.5-7B-Instruct
)
DATASET_NAMES=(
  commonsense_qa
  commonsense_qa
  arc_challenge
  arc_challenge
)
AYS_MC_DATASETS_BY_TASK=(
  commonsense_qa
  commonsense_qa
  arc_challenge
  arc_challenge
)
RUN_SLUGS=(
  commonsense_qa_llama31_8b
  commonsense_qa_qwen25_7b
  arc_challenge_llama31_8b
  arc_challenge_qwen25_7b
)
REQUEST_STRENGTHS=(
  weak
  strong
)

SELECTED_INDICES=()
for index in "${!TASK_LABELS[@]}"; do
  if [[ -z "$TASK_FILTER" || "$TASK_FILTER" == "${TASK_LABELS[$index]}" ]]; then
    SELECTED_INDICES+=("$index")
  fi
done
if [[ "${#SELECTED_INDICES[@]}" -eq 0 ]]; then
  printf '%s\n' "TASK_FILTER did not match a known task label: $TASK_FILTER" >&2
  exit 1
fi

SELECTED_REQUESTS=()
for request_strength in "${REQUEST_STRENGTHS[@]}"; do
  if [[ -z "$REQUEST_FILTER" || "$REQUEST_FILTER" == "$request_strength" ]]; then
    SELECTED_REQUESTS+=("$request_strength")
  fi
done
if [[ "${#SELECTED_REQUESTS[@]}" -eq 0 ]]; then
  printf '%s\n' "REQUEST_FILTER did not match weak or strong: $REQUEST_FILTER" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS=$(( ${#SELECTED_INDICES[@]} * ${#SELECTED_REQUESTS[@]} ))
if (( TASK_ID >= TOTAL_TASKS )); then
  printf '[%s] skipping out-of-range task_id=%s total_tasks=%s\n' \
    "$SCRIPT_TAG" "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi

COMBO_OFFSET=$(( TASK_ID / ${#SELECTED_REQUESTS[@]} ))
REQUEST_OFFSET=$(( TASK_ID % ${#SELECTED_REQUESTS[@]} ))
BASE_INDEX="${SELECTED_INDICES[$COMBO_OFFSET]}"
REQUEST_STRENGTH="${SELECTED_REQUESTS[$REQUEST_OFFSET]}"

TASK_LABEL="${TASK_LABELS[$BASE_INDEX]}"
MODEL_ID="${MODEL_IDS[$BASE_INDEX]}"
DATASET_NAME="${DATASET_NAMES[$BASE_INDEX]}"
AYS_MC_DATASETS="${AYS_MC_DATASETS_BY_TASK[$BASE_INDEX]}"
RUN_NAME="${RUN_SLUGS[$BASE_INDEX]}_antisyc_${REQUEST_STRENGTH}_sampling_${REQUEST_TAG}"

slugify() {
  "$ENV_PYTHON" - "$1" <<'PY'
import sys
value = sys.argv[1]
cleaned = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
print(cleaned or "model")
PY
}

iso_now() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

verify_sampling_run() {
  local run_dir="$1"
  local expected_request="$2"
  "$ENV_PYTHON" - "$run_dir" "$expected_request" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_request = sys.argv[2]
required = [
    run_dir / "meta" / "run_config.json",
    run_dir / "meta" / "status.json",
    run_dir / "meta" / "run_summary.json",
    run_dir / "meta" / "run_manifest.json",
    run_dir / "runtime" / "logs" / "run.log",
    run_dir / "sampling" / "raw" / "sampling_manifest.json",
    run_dir / "sampling" / "flat" / "sampled_responses.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required sampling artifacts:\n" + "\n".join(missing))

run_config = json.loads((run_dir / "meta" / "run_config.json").read_text(encoding="utf-8"))
status = json.loads((run_dir / "meta" / "status.json").read_text(encoding="utf-8"))
sampling_manifest = json.loads((run_dir / "sampling" / "raw" / "sampling_manifest.json").read_text(encoding="utf-8"))

if status.get("status") != "completed":
    raise SystemExit(f"run status is not completed: {status.get('status')!r}")
if not bool(run_config.get("sampling_only", False)):
    raise SystemExit("request sampling shard expected sampling_only=true")
actual_request = str(run_config.get("anti_sycophancy_request", "") or "none")
if actual_request != expected_request:
    raise SystemExit(f"anti_sycophancy_request mismatch: {actual_request!r} != {expected_request!r}")
if str(sampling_manifest.get("sampling_hash", "") or "") == "":
    raise SystemExit("sampling manifest did not record a sampling hash")

with (run_dir / "sampling" / "flat" / "sampled_responses.csv").open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
if not rows:
    raise SystemExit("sampled_responses.csv is empty")
seen_requests = {str(row.get("anti_sycophancy_request", "") or "none") for row in rows}
if seen_requests != {expected_request}:
    raise SystemExit(f"sample rows contain unexpected request labels: {sorted(seen_requests)}")

print(
    "[postcheck] anti_sycophancy_sampling_ok"
    f" run_dir={run_dir}"
    f" request={expected_request}"
    f" sample_rows={len(rows)}"
    f" sampling_hash={sampling_manifest.get('sampling_hash')}"
)
PY
}

MODEL_SLUG="$(slugify "$MODEL_ID")"
RUN_DIR="$OUT_DIR/$MODEL_SLUG/$DATASET_NAME/$RUN_NAME"

STRUCTURED_LOG_DIR="$LOG_ROOT/by_task/$TASK_LABEL/$REQUEST_STRENGTH/sampling/job_${SLURM_JOB_ID:-local}"
mkdir -p "$STRUCTURED_LOG_DIR"
STRUCTURED_STDOUT="$STRUCTURED_LOG_DIR/task_${TASK_ID}.out"
STRUCTURED_STDERR="$STRUCTURED_LOG_DIR/task_${TASK_ID}.err"
exec > >(tee -a "$STRUCTURED_STDOUT") 2> >(tee -a "$STRUCTURED_STDERR" >&2)

if [[ "$ALLOW_STALE_LOCK_CLEANUP" == "1" && -f "$RUN_DIR/.run.lock" ]]; then
  printf '%s\n' "[$SCRIPT_TAG] removing stale lock by request: $RUN_DIR/.run.lock"
  rm -f "$RUN_DIR/.run.lock"
fi

cmd=(
  "$ENV_PYTHON" run_sycophancy_bias_probe.py
  --model "$MODEL_ID"
  --device "$DEVICE"
  --hf_cache_dir "$HF_CACHE_DIR"
  --benchmark_source ays_mc_single_turn
  --input_jsonl are_you_sure.jsonl
  --dataset_name "$DATASET_NAME"
  --ays_mc_datasets "$AYS_MC_DATASETS"
  --mc_mode strict_mc
  --bias_types "$BIAS_TYPES_CSV"
  --anti_sycophancy_request "$REQUEST_STRENGTH"
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --split_seed "$SPLIT_SEED"
  --seed "$SEED"
  --probe_seed "$PROBE_SEED"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --sampling_only
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
)

if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
  cmd+=(--device_map_auto)
fi
if [[ -n "$MAX_QUESTIONS" ]]; then
  cmd+=(--max_questions "$MAX_QUESTIONS")
fi

printf '[%s] task=%s request=%s model=%s dataset=%s\n' "$SCRIPT_TAG" "$TASK_LABEL" "$REQUEST_STRENGTH" "$MODEL_ID" "$DATASET_NAME"
printf '[%s] started_at=%s hostname=%s pwd=%s\n' \
  "$SCRIPT_TAG" "$(iso_now)" "$(hostname)" "$PWD"
printf '[%s] slurm job_id=%s array_job_id=%s array_task_id=%s cpus=%s mem_per_node=%s\n' \
  "$SCRIPT_TAG" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-0}" \
  "${SLURM_CPUS_PER_TASK:-unknown}" "${SLURM_MEM_PER_NODE:-unknown}"
printf '[%s] bias_types=%s\n' "$SCRIPT_TAG" "$BIAS_TYPES_CSV"
printf '[%s] run_name=%s\n' "$SCRIPT_TAG" "$RUN_NAME"
printf '[%s] run_dir=%s\n' "$SCRIPT_TAG" "$RUN_DIR"
printf '[%s] structured_stdout=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDOUT"
printf '[%s] structured_stderr=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDERR"
sycophancy_bias_print_storage_env "[$SCRIPT_TAG]"
printf '[%s] memory_env: SAMPLE_BATCH_SIZE=%s OMP_NUM_THREADS=%s MALLOC_ARENA_MAX=%s TOKENIZERS_PARALLELISM=%s PYTORCH_CUDA_ALLOC_CONF=%s\n' \
  "$SCRIPT_TAG" "$SAMPLE_BATCH_SIZE" "$OMP_NUM_THREADS" "$MALLOC_ARENA_MAX" "$TOKENIZERS_PARALLELISM" "$PYTORCH_CUDA_ALLOC_CONF"
if command -v nvidia-smi >/dev/null 2>&1; then
  printf '[%s] nvidia-smi at start:\n' "$SCRIPT_TAG"
  nvidia-smi || true
fi
printf '[%s] command: ' "$SCRIPT_TAG"
printf '%q ' "${cmd[@]}"
printf '\n'

START_EPOCH="$(date +%s)"
set +e
"${cmd[@]}"
STATUS="$?"
set -e

if [[ "$STATUS" -eq 0 ]]; then
  printf '[%s] verifying sampling artifacts...\n' "$SCRIPT_TAG"
  set +e
  verify_sampling_run "$RUN_DIR" "$REQUEST_STRENGTH"
  VERIFY_STATUS="$?"
  set -e
  if [[ "$VERIFY_STATUS" -ne 0 ]]; then
    STATUS="$VERIFY_STATUS"
  fi
fi

END_EPOCH="$(date +%s)"
printf '[%s] finished_at=%s exit_status=%s elapsed_seconds=%s\n' \
  "$SCRIPT_TAG" "$(iso_now)" "$STATUS" "$((END_EPOCH - START_EPOCH))"
if command -v nvidia-smi >/dev/null 2>&1; then
  printf '[%s] nvidia-smi at end:\n' "$SCRIPT_TAG"
  nvidia-smi || true
fi
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sstat >/dev/null 2>&1; then
  printf '[%s] sstat snapshot:\n' "$SCRIPT_TAG"
  sstat --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU -j "${SLURM_JOB_ID}.batch" || true
fi
exit "$STATUS"
