#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p jobs/sycophancy_bias_probe/logs

sbatch jobs/sycophancy_bias_probe/backfill_bias_probes_on_neutral_20260402_seas.sbatch "$@"
