#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

sbatch jobs/sycophancy_bias_probe/fast_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/medium_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc.sbatch
