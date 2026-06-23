#!/bin/bash
#SBATCH --job-name=syco_antisyc_eval_20260623
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/analysis/%x.%A_%a.out
#SBATCH --error=/n/holystore01/LABS/barak_lab/Users/ishapira/LLMsKnow_logs/sycophancy_bias_probe/anti_sycophancy_request_sharded_20260623/slurm/analysis/%x.%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

BUNDLE_NAME="anti_sycophancy_request_sharded_20260623"
SCRIPT_TAG="anti-sycophancy-analysis-20260623"
BASELINE_TAG="${BASELINE_TAG:-20260618}"
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
  "$LOG_ROOT/slurm/analysis" \
  "$LOG_ROOT/by_task"

DEVICE="${DEVICE:-auto}"
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-test}"
MAX_PROBE_PAIRS="${MAX_PROBE_PAIRS:-}"
TASK_FILTER="${TASK_FILTER:-}"
REQUEST_FILTER="${REQUEST_FILTER:-}"

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
RUN_SLUG="${RUN_SLUGS[$BASE_INDEX]}"

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

upper_env_key() {
  printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_'
}

assert_sampling_run() {
  local run_dir="$1"
  local expected_request="$2"
  local label="$3"
  "$ENV_PYTHON" - "$run_dir" "$expected_request" "$label" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_request = sys.argv[2]
label = sys.argv[3]
required = [
    run_dir / "meta" / "run_config.json",
    run_dir / "meta" / "status.json",
    run_dir / "sampling" / "raw" / "sampling_manifest.json",
    run_dir / "sampling" / "raw" / "sampling_records.jsonl",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"{label}: missing required sampling artifacts:\n" + "\n".join(missing))
status = json.loads((run_dir / "meta" / "status.json").read_text(encoding="utf-8"))
run_config = json.loads((run_dir / "meta" / "run_config.json").read_text(encoding="utf-8"))
if status.get("status") != "completed":
    raise SystemExit(f"{label}: run status is not completed: {status.get('status')!r}")
actual_request = str(run_config.get("anti_sycophancy_request", "") or "none")
if actual_request != expected_request:
    raise SystemExit(f"{label}: request mismatch {actual_request!r} != {expected_request!r}")
PY
}

assert_probe_run() {
  local run_dir="$1"
  "$ENV_PYTHON" - "$run_dir" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidates = [
    run_dir / "probes" / "chosen" / "families" / "probe_bias_random_all",
    run_dir / "probes" / "chosen_probe" / "probe_bias_random_all",
]
for candidate in candidates:
    if (candidate / "model.pkl").exists() and (candidate / "metadata.json").exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(
    "missing chosen probe_bias_random_all artifacts. Tried:\n"
    + "\n".join(str(candidate) for candidate in candidates)
)
PY
}

verify_analysis_output() {
  local output_dir="$1"
  "$ENV_PYTHON" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
required = [
    output_dir / "metadata.json",
    output_dir / "behavior_neutral_stability_items.csv",
    output_dir / "behavior_family_mitigation_items.csv",
    output_dir / "behavior_summary.csv",
    output_dir / "random_all_probe_choice_scores.csv",
    output_dir / "random_all_probe_choice_deltas.csv",
    output_dir / "random_all_probe_prompt_movement_items.csv",
    output_dir / "random_all_probe_summary.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required comparison outputs:\n" + "\n".join(missing))
metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
if int(metadata.get("n_neutral_stability_rows", 0) or 0) <= 0:
    raise SystemExit("comparison metadata reports no neutral stability rows")
if int(metadata.get("n_family_mitigation_rows", 0) or 0) <= 0:
    raise SystemExit("comparison metadata reports no family mitigation rows")
if int(metadata.get("n_probe_matched_pairs", 0) or 0) <= 0:
    raise SystemExit("comparison metadata reports no matched probe pairs")
print(
    "[postcheck] anti_sycophancy_analysis_ok"
    f" output_dir={output_dir}"
    f" neutral_rows={metadata.get('n_neutral_stability_rows')}"
    f" family_rows={metadata.get('n_family_mitigation_rows')}"
    f" probe_pairs={metadata.get('n_probe_matched_pairs')}"
)
PY
}

MODEL_SLUG="$(slugify "$MODEL_ID")"
REQUEST_RUN_NAME="${RUN_SLUG}_antisyc_${REQUEST_STRENGTH}_sampling_${REQUEST_TAG}"
DEFAULT_REQUEST_RUN_DIR="$OUT_DIR/$MODEL_SLUG/$DATASET_NAME/$REQUEST_RUN_NAME"
DEFAULT_BASELINE_SAMPLING_RUN_DIR="$OUT_DIR/$MODEL_SLUG/$DATASET_NAME/${RUN_SLUG}_allfamilies_sampling_${BASELINE_TAG}"
DEFAULT_BASELINE_RANDOM_ALL_PROBE_RUN_DIR="$OUT_DIR/$MODEL_SLUG/$DATASET_NAME/${RUN_SLUG}_allfamilies_probe_random_all_${BASELINE_TAG}"
COMPARISON_DATASET_MODEL="${DATASET_NAME}_${MODEL_SLUG}"

TASK_ENV_KEY="$(upper_env_key "$TASK_LABEL")"
SPECIFIC_BASELINE_SAMPLING_VAR="BASELINE_SAMPLING_RUN_DIR_${TASK_ENV_KEY}"
SPECIFIC_BASELINE_PROBE_VAR="BASELINE_RANDOM_ALL_PROBE_RUN_DIR_${TASK_ENV_KEY}"
SPECIFIC_REQUEST_VAR="REQUEST_RUN_DIR_${TASK_ENV_KEY}_${REQUEST_STRENGTH^^}"
SPECIFIC_BASELINE_SAMPLING="${!SPECIFIC_BASELINE_SAMPLING_VAR-}"
SPECIFIC_BASELINE_PROBE="${!SPECIFIC_BASELINE_PROBE_VAR-}"
SPECIFIC_REQUEST_RUN="${!SPECIFIC_REQUEST_VAR-}"

REQUEST_RUN_DIR="${SPECIFIC_REQUEST_RUN:-${REQUEST_RUN_DIR:-$DEFAULT_REQUEST_RUN_DIR}}"
BASELINE_SAMPLING_RUN_DIR="${SPECIFIC_BASELINE_SAMPLING:-${BASELINE_SAMPLING_RUN_DIR:-$DEFAULT_BASELINE_SAMPLING_RUN_DIR}}"
BASELINE_RANDOM_ALL_PROBE_RUN_DIR="${SPECIFIC_BASELINE_PROBE:-${BASELINE_RANDOM_ALL_PROBE_RUN_DIR:-$DEFAULT_BASELINE_RANDOM_ALL_PROBE_RUN_DIR}}"
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR:-$OUT_DIR/_comparisons/anti_sycophancy_request_20260623/$COMPARISON_DATASET_MODEL/$REQUEST_STRENGTH}"

STRUCTURED_LOG_DIR="$LOG_ROOT/by_task/$TASK_LABEL/$REQUEST_STRENGTH/analysis/job_${SLURM_JOB_ID:-local}"
mkdir -p "$STRUCTURED_LOG_DIR"
STRUCTURED_STDOUT="$STRUCTURED_LOG_DIR/task_${TASK_ID}.out"
STRUCTURED_STDERR="$STRUCTURED_LOG_DIR/task_${TASK_ID}.err"
exec > >(tee -a "$STRUCTURED_STDOUT") 2> >(tee -a "$STRUCTURED_STDERR" >&2)

printf '[%s] task=%s request=%s model=%s dataset=%s split=%s\n' \
  "$SCRIPT_TAG" "$TASK_LABEL" "$REQUEST_STRENGTH" "$MODEL_ID" "$DATASET_NAME" "$ANALYSIS_SPLIT"
printf '[%s] started_at=%s hostname=%s pwd=%s\n' \
  "$SCRIPT_TAG" "$(iso_now)" "$(hostname)" "$PWD"
printf '[%s] slurm job_id=%s array_job_id=%s array_task_id=%s cpus=%s mem_per_node=%s\n' \
  "$SCRIPT_TAG" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-0}" \
  "${SLURM_CPUS_PER_TASK:-unknown}" "${SLURM_MEM_PER_NODE:-unknown}"
printf '[%s] request_run_dir=%s\n' "$SCRIPT_TAG" "$REQUEST_RUN_DIR"
printf '[%s] baseline_sampling_run_dir=%s\n' "$SCRIPT_TAG" "$BASELINE_SAMPLING_RUN_DIR"
printf '[%s] baseline_random_all_probe_run_dir=%s\n' "$SCRIPT_TAG" "$BASELINE_RANDOM_ALL_PROBE_RUN_DIR"
printf '[%s] comparison_output_dir=%s\n' "$SCRIPT_TAG" "$COMPARISON_OUTPUT_DIR"
printf '[%s] structured_stdout=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDOUT"
printf '[%s] structured_stderr=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDERR"
sycophancy_bias_print_storage_env "[$SCRIPT_TAG]"
printf '[%s] memory_env: OMP_NUM_THREADS=%s MALLOC_ARENA_MAX=%s TOKENIZERS_PARALLELISM=%s PYTORCH_CUDA_ALLOC_CONF=%s\n' \
  "$SCRIPT_TAG" "$OMP_NUM_THREADS" "$MALLOC_ARENA_MAX" "$TOKENIZERS_PARALLELISM" "$PYTORCH_CUDA_ALLOC_CONF"

assert_sampling_run "$BASELINE_SAMPLING_RUN_DIR" "none" "baseline sampling"
assert_sampling_run "$REQUEST_RUN_DIR" "$REQUEST_STRENGTH" "request sampling"
PROBE_DIR="$(assert_probe_run "$BASELINE_RANDOM_ALL_PROBE_RUN_DIR")"
printf '[%s] resolved_probe_dir=%s\n' "$SCRIPT_TAG" "$PROBE_DIR"

cmd=(
  "$ENV_PYTHON" scripts/evaluate_anti_sycophancy_request.py
  --baseline_run_dir "$BASELINE_SAMPLING_RUN_DIR"
  --request_run_dir "$REQUEST_RUN_DIR"
  --probe_run_dir "$BASELINE_RANDOM_ALL_PROBE_RUN_DIR"
  --split "$ANALYSIS_SPLIT"
  --hf_cache_dir "$HF_CACHE_DIR"
  --device "$DEVICE"
  --output_dir "$COMPARISON_OUTPUT_DIR"
)

if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
  cmd+=(--device_map_auto)
fi
if [[ -n "$MAX_PROBE_PAIRS" ]]; then
  cmd+=(--max_probe_pairs "$MAX_PROBE_PAIRS")
fi

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
  printf '[%s] verifying comparison outputs...\n' "$SCRIPT_TAG"
  set +e
  verify_analysis_output "$COMPARISON_OUTPUT_DIR"
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
