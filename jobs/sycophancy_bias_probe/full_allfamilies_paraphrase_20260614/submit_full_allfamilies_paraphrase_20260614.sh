#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614

SBATCH_ARRAY="${SBATCH_ARRAY:-0-3}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_MEM="${SBATCH_MEM:-100G}"
SBATCH_CPUS="${SBATCH_CPUS:-2}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu,seas_gpu,gpu_h200}"

SBATCH_CMD=(
  sbatch
  --array "$SBATCH_ARRAY"
  --time "$SBATCH_TIME"
  --mem "$SBATCH_MEM"
  --cpus-per-task "$SBATCH_CPUS"
  --partition "$SBATCH_PARTITION"
  jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/full_allfamilies_paraphrase_20260614_array.sbatch
)

printf '[submit-full-allfamilies-paraphrase-20260614] '
printf '%q ' "${SBATCH_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
