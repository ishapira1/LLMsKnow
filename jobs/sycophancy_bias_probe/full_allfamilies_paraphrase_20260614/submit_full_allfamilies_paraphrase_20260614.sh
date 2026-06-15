#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs/full_allfamilies_paraphrase_20260614

sbatch jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_20260614/full_allfamilies_paraphrase_20260614_array.sbatch
