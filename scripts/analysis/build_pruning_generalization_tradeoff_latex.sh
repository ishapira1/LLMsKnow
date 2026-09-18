#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
source_dir="$repo_root/scripts/analysis/latex_pruning_generalization"
build_dir="$repo_root/tmp/pdfs/pruning_generalization_latex"
output_dir="$repo_root/output/pdf"

mkdir -p "$build_dir" "$output_dir"
cp "$source_dir"/*.tex "$build_dir"/

documents=(
  pruning_generalization_latex_panel_a
  pruning_generalization_latex_panel_b
  pruning_generalization_latex_panel_c
  pruning_generalization_latex_legend
  pruning_generalization_latex_figure
)

for document in "${documents[@]}"; do
  (
    cd "$build_dir"
    /opt/homebrew/bin/tectonic -C --keep-logs "$document.tex"
  )
  cp "$build_dir/$document.pdf" "$output_dir/$document.pdf"
done

