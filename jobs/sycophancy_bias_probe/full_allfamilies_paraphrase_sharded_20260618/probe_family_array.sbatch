#!/bin/bash
#SBATCH --job-name=syco_allfam_probe_20260618
#SBATCH --partition=gpu,seas_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-47
#SBATCH --output=jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260618/slurm/probes/%x.%A_%a.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_sharded_20260618/slurm/probes/%x.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=itaishapira@g.harvard.edu

set -euo pipefail

BUNDLE_NAME="full_allfamilies_paraphrase_sharded_20260618"
SCRIPT_TAG="probe-family-sharded-20260618"
REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
LOG_ROOT="jobs/sycophancy_bias_probe/logs/${BUNDLE_NAME}"

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

HF_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-}}"
if [[ -z "$HF_CACHE_DIR" ]]; then
  printf '%s\n' "HUGGINGFACE_HUB_CACHE or HF_HUB_CACHE must be set in .env" >&2
  exit 1
fi
if [[ "$HF_CACHE_DIR" == /home/* ]]; then
  printf '%s\n' "Refusing to run: HF cache points to /home ($HF_CACHE_DIR)" >&2
  exit 1
fi

HF_DATASETS_DIR="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
HF_HOME_DIR="${HF_HOME:-$(dirname "$HF_CACHE_DIR")/hf_home}"
export HF_HUB_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_DIR"
export HF_HOME="$HF_HOME_DIR"

mkdir -p \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_HOME" \
  "$LOG_ROOT" \
  "$LOG_ROOT/slurm/probes" \
  "$LOG_ROOT/by_task"

DEFAULT_BIAS_TYPES_CSV="$("$ENV_PYTHON" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=False)))')"
DEFAULT_PROBE_FAMILIES_CSV="$("$ENV_PYTHON" -c 'from llmssycoph.data import trainable_prompt_families; print(",".join(trainable_prompt_families(include_neutral=True)))')"
BIAS_TYPES_CSV="${BIAS_TYPES_CSV:-$DEFAULT_BIAS_TYPES_CSV}"
PROBE_FAMILIES_CSV="${PROBE_FAMILIES_CSV:-$DEFAULT_PROBE_FAMILIES_CSV}"
PROBE_FAMILY_FILTER="${PROBE_FAMILY_FILTER:-}"
if [[ -n "$PROBE_FAMILY_FILTER" ]]; then
  PROBE_FAMILIES_CSV="$PROBE_FAMILY_FILTER"
fi
PARAPHRASE_ARTIFACT_PATH="${PARAPHRASE_ARTIFACT_PATH:-data/ad_hoc/paraphrase_robustness_test_stems_v1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
OUT_DIR="${OUT_DIR:-results/sycophancy_bias_probe}"
SPLIT_SEED="${SPLIT_SEED:-5}"
SEED="${SEED:-5}"
PROBE_SEED="${PROBE_SEED:-5}"
PROBE_LAYER_MIN="${PROBE_LAYER_MIN:-1}"
PROBE_LAYER_MAX="${PROBE_LAYER_MAX:-999}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PROBE_SELECTION_MAX_SAMPLES="${PROBE_SELECTION_MAX_SAMPLES:-}"
PROBE_TRAIN_MAX_SAMPLES="${PROBE_TRAIN_MAX_SAMPLES:-}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
DEVICE_MAP_AUTO="${DEVICE_MAP_AUTO:-0}"
ALLOW_STALE_LOCK_CLEANUP="${ALLOW_STALE_LOCK_CLEANUP:-0}"
TASK_FILTER="${TASK_FILTER:-}"

if [[ -z "$BIAS_TYPES_CSV" ]]; then
  printf '%s\n' "No trainable non-neutral prompt families resolved." >&2
  exit 1
fi
if [[ -z "$PROBE_FAMILIES_CSV" ]]; then
  printf '%s\n' "No probe families resolved." >&2
  exit 1
fi
if [[ ! -d "$PARAPHRASE_ARTIFACT_PATH" && ! -f "$PARAPHRASE_ARTIFACT_PATH" ]]; then
  printf '%s\n' "Missing paraphrase artifact path: $PARAPHRASE_ARTIFACT_PATH" >&2
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

IFS=, read -r -a PROBE_FAMILIES <<< "$PROBE_FAMILIES_CSV"
if [[ "${#PROBE_FAMILIES[@]}" -eq 0 ]]; then
  printf '%s\n' "No probe families parsed from PROBE_FAMILIES_CSV." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS=$(( ${#SELECTED_INDICES[@]} * ${#PROBE_FAMILIES[@]} ))
if (( TASK_ID >= TOTAL_TASKS )); then
  printf '[%s] skipping out-of-range task_id=%s total_tasks=%s\n' \
    "$SCRIPT_TAG" "$TASK_ID" "$TOTAL_TASKS"
  exit 0
fi

COMBO_OFFSET=$(( TASK_ID / ${#PROBE_FAMILIES[@]} ))
FAMILY_OFFSET=$(( TASK_ID % ${#PROBE_FAMILIES[@]} ))
BASE_INDEX="${SELECTED_INDICES[$COMBO_OFFSET]}"
PROBE_FAMILY="${PROBE_FAMILIES[$FAMILY_OFFSET]}"

TASK_LABEL="${TASK_LABELS[$BASE_INDEX]}"
MODEL_ID="${MODEL_IDS[$BASE_INDEX]}"
DATASET_NAME="${DATASET_NAMES[$BASE_INDEX]}"
AYS_MC_DATASETS="${AYS_MC_DATASETS_BY_TASK[$BASE_INDEX]}"
RUN_NAME="${RUN_SLUGS[$BASE_INDEX]}_allfamilies_probe_${PROBE_FAMILY}_20260618"

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

verify_probe_run() {
  local run_dir="$1"
  "$ENV_PYTHON" - "$run_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
required = [
    run_dir / "meta" / "run_config.json",
    run_dir / "meta" / "status.json",
    run_dir / "meta" / "run_summary.json",
    run_dir / "meta" / "run_manifest.json",
    run_dir / "runtime" / "logs" / "run.log",
    run_dir / "query" / "artifact_catalog.jsonl",
    run_dir / "query" / "chosen_probe_registry.csv",
    run_dir / "query" / "chosen_probe_metrics.csv",
    run_dir / "query" / "chosen_probe_cross_family_metrics.csv",
    run_dir / "query" / "chosen_probe_movement_summary.csv",
    run_dir / "query" / "chosen_probe_movement_items.jsonl",
    run_dir / "query" / "paraphrase_coverage.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required probe artifacts:\n" + "\n".join(missing))

run_config = json.loads((run_dir / "meta" / "run_config.json").read_text(encoding="utf-8"))
status = json.loads((run_dir / "meta" / "status.json").read_text(encoding="utf-8"))
run_manifest = json.loads((run_dir / "meta" / "run_manifest.json").read_text(encoding="utf-8"))

if status.get("status") != "completed":
    raise SystemExit(f"run status is not completed: {status.get('status')!r}")
if bool(run_config.get("sampling_only", False)):
    raise SystemExit("probe shard expected sampling_only=false")

features = dict(run_manifest.get("features", {}) or {})
if not bool(features.get("paraphrase_movement_present", False)):
    raise SystemExit("run_manifest reports paraphrase_movement_present=false")
if not bool(features.get("random_baseline_geometry_present", False)):
    raise SystemExit("run_manifest reports random_baseline_geometry_present=false")

def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))

registry_rows = count_csv_rows(run_dir / "query" / "chosen_probe_registry.csv")
metrics_rows = count_csv_rows(run_dir / "query" / "chosen_probe_metrics.csv")
cross_family_rows = count_csv_rows(run_dir / "query" / "chosen_probe_cross_family_metrics.csv")
movement_summary_rows = count_csv_rows(run_dir / "query" / "chosen_probe_movement_summary.csv")
paraphrase_coverage_rows = count_csv_rows(run_dir / "query" / "paraphrase_coverage.csv")
movement_item_rows = sum(
    1
    for line in (run_dir / "query" / "chosen_probe_movement_items.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
)

for label, value in {
    "chosen_probe_registry.csv": registry_rows,
    "chosen_probe_metrics.csv": metrics_rows,
    "chosen_probe_cross_family_metrics.csv": cross_family_rows,
    "chosen_probe_movement_summary.csv": movement_summary_rows,
    "chosen_probe_movement_items.jsonl": movement_item_rows,
    "paraphrase_coverage.csv": paraphrase_coverage_rows,
}.items():
    if value <= 0:
        raise SystemExit(f"{label} is empty")

print(
    "[postcheck] probe_ok"
    f" run_dir={run_dir}"
    f" chosen_probe_rows={metrics_rows}"
    f" cross_family_rows={cross_family_rows}"
    f" movement_summary_rows={movement_summary_rows}"
    f" movement_item_rows={movement_item_rows}"
    f" paraphrase_coverage_rows={paraphrase_coverage_rows}"
)
PY
}

MODEL_SLUG="$(slugify "$MODEL_ID")"
RUN_DIR="$OUT_DIR/$MODEL_SLUG/$DATASET_NAME/$RUN_NAME"

STRUCTURED_LOG_DIR="$LOG_ROOT/by_task/$TASK_LABEL/probe_$PROBE_FAMILY/job_${SLURM_JOB_ID:-local}"
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
  --probe_families "$PROBE_FAMILY"
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --split_seed "$SPLIT_SEED"
  --seed "$SEED"
  --probe_seed "$PROBE_SEED"
  --probe_layer_min "$PROBE_LAYER_MIN"
  --probe_layer_max "$PROBE_LAYER_MAX"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --paraphrase_artifact_path "$PARAPHRASE_ARTIFACT_PATH"
  --run_name "$RUN_NAME"
  --out_dir "$OUT_DIR"
)

if [[ "$DEVICE_MAP_AUTO" == "1" ]]; then
  cmd+=(--device_map_auto)
fi
if [[ -n "$PROBE_SELECTION_MAX_SAMPLES" ]]; then
  cmd+=(--probe_selection_max_samples "$PROBE_SELECTION_MAX_SAMPLES")
fi
if [[ -n "$PROBE_TRAIN_MAX_SAMPLES" ]]; then
  cmd+=(--probe_train_max_samples "$PROBE_TRAIN_MAX_SAMPLES")
fi
if [[ -n "$MAX_QUESTIONS" ]]; then
  cmd+=(--max_questions "$MAX_QUESTIONS")
fi

printf '[%s] task=%s probe_family=%s model=%s dataset=%s\n' \
  "$SCRIPT_TAG" "$TASK_LABEL" "$PROBE_FAMILY" "$MODEL_ID" "$DATASET_NAME"
printf '[%s] started_at=%s hostname=%s pwd=%s\n' \
  "$SCRIPT_TAG" "$(iso_now)" "$(hostname)" "$PWD"
printf '[%s] slurm job_id=%s array_job_id=%s array_task_id=%s cpus=%s mem_per_node=%s total_tasks=%s\n' \
  "$SCRIPT_TAG" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-0}" \
  "${SLURM_CPUS_PER_TASK:-unknown}" "${SLURM_MEM_PER_NODE:-unknown}" "$TOTAL_TASKS"
printf '[%s] bias_types=%s\n' "$SCRIPT_TAG" "$BIAS_TYPES_CSV"
printf '[%s] paraphrase_artifact_path=%s\n' "$SCRIPT_TAG" "$PARAPHRASE_ARTIFACT_PATH"
printf '[%s] run_name=%s\n' "$SCRIPT_TAG" "$RUN_NAME"
printf '[%s] run_dir=%s\n' "$SCRIPT_TAG" "$RUN_DIR"
printf '[%s] structured_stdout=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDOUT"
printf '[%s] structured_stderr=%s\n' "$SCRIPT_TAG" "$STRUCTURED_STDERR"
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
  printf '[%s] verifying probe artifacts...\n' "$SCRIPT_TAG"
  set +e
  verify_probe_run "$RUN_DIR"
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
