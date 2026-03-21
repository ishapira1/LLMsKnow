#!/bin/bash
#SBATCH --job-name=full_aqua_mc_gpt54nano_20260320_seas
#SBATCH --partition=shared,sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=jobs/sycophancy_bias_probe/logs/%x.%j.out
#SBATCH --error=jobs/sycophancy_bias_probe/logs/%x.%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/n/home12/ishapira/LLMsKnow}"
export MODEL_ID="${MODEL_ID:-gpt-5.4-nano}"
export RUN_NAME="${RUN_NAME:-full_aqua_mc_gpt54nano_20260320_auto_allq_samples_seas}"

exec "$REPO_DIR/jobs/sycophancy_bias_probe/full_aqua_mc_gpt54nano_samples.sh" "$@"
