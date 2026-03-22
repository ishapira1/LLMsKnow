# Notebook Blueprint

This file documents how analysis notebooks should be structured so later edits stay consistent.

## Top-Level Principles

- The notebook is a report, not a scratchpad.
- Use one cell per output artifact whenever possible.
- Separate external-behavior analysis from probe analysis.
- Keep label-free and label-dependent claims clearly separated.
- Never silently replace unavailable quantities with a proxy. If a quantity is unavailable from saved artifacts, say so explicitly.

## Required Notebook Flow

1. Title and short orientation
2. Setup / context loading cell
3. Section markdown
4. Subsection markdown with goal
5. One output cell per figure or table
6. Optional short interpretation markdown

## Section And Subsection Rules

Each subsection should state:

- what question it answers
- what notation is used
- what the reader should look for

Preferred structure:

- `Goal`
- `Quantity`
- `How to read`

Keep these short. The notebook should remain easy to skim.

## Cell Rules

- One figure or one table per cell.
- Do not combine unrelated outputs in one cell.
- All plot/table cells must save their artifacts to the run's `analysis/` subtree.
- All plot/table cells must go through the safe execution wrapper.

## Plotting Rules

- Use `seaborn.set_style("white")`
- Titles should be large
- Axis label fontsize should be at least `15`
- Tick label fontsize should be at least `12`
- Legends should be below the plot
- Use consistent predefined colors across related analyses
- Keep axis limits aligned across sibling panels when possible
- Put `n` in panel titles when that helps interpretation
- Use bootstrap confidence intervals for summary curves and bars when requested by the spec

## Naming Rules

Use deterministic filenames based on section order and content.

Examples:

- `01_01_run_overview.csv`
- `01_02_accuracy_by_template.csv`
- `01_03_effective_num_responses_histogram.pdf`
- `02_01_probe_auc_by_layer.pdf`

When a cleaner semantic filename is more helpful, keep it short and stable.

## Notation Rules

Prefer explicit mathematical notation over prose labels.

Examples:

- `x` = neutral prompt
- `x'` = bias-injected prompt
- `c` = correct answer
- `b` = bias target
- `ŷ_x` = chosen answer under `x`
- `s(x, a)` = probe score for answer `a`

Avoid vague labels like "external drop" when a clear delta notation is available.

## Availability Rules

Each analysis function should declare whether it needs:

- probes
- correctness labels
- explicit bias targets
- candidate-level scores

If those are missing:

- the notebook cell should render a short warning
- the failure should be recorded in `analysis/tables/analysis_cell_failures.csv`
- later cells should still run

## Future AI Editing Rules

When extending a notebook:

1. add reusable logic to Python modules first
2. do not paste large data-wrangling blocks directly into notebook JSON
3. add or update the spec entry
4. keep markdown narrative aligned with the actual metric definitions
5. prefer reusing notation from the section header instead of inventing new synonyms

If a section is complex, add a short markdown roadmap at the section start.

## Current Design Intent

The framework is meant to support:

- repeated notebook generation for many runs
- ad-hoc notebooks for a single run
- future AI edits without structural drift

So the code should optimize for:

- explicitness
- small reusable functions
- deterministic outputs
- graceful failure
