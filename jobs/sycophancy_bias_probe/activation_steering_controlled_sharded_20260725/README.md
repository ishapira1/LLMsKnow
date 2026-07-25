# Controlled Prompt-Only Activation Steering — 2026-07-25

This bundle implements `controlled_prompt_only_v1_20260725`. It is separate from
the legacy random-all restoration experiment because its direction sign, prompt
contrast, aggregation, and alpha units are different.

## Hard gates

- The eight-row manifest may be inspected while review status is `pending`.
- Fitting or intervention requires every included row to have
  `semantic_b_review_status="approved"`.
- Same-shape alpha zero must be bitwise exact.
- Cross-batch top-choice agreement must be 100%, maximum option-probability
  error at most 0.005, and c-b margin error at most 0.05.
- A failed batch-shape replay forces treatment batching to one and is recorded
  in the tiny compute projection.
- The full submitter requires an audited 1,000-row manifest, the fixed Alpaca
  manifest, an eight-example inspection report, a reviewed tiny compute report,
  a clean approved Git commit, a hash-bound researcher approval JSON, and the
  explicit `ALLOW_FULL_SUBMISSION=1` opt-in.
- No script removes `.run.lock` or output artifacts.

## Inspection and tiny run

```bash
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
EXPERIMENT_RUN_ID=activation_steering_examples_qwen_csqa_20260725_v1 \
sbatch jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/inspect_examples_qwen_csqa.sbatch
```

After researcher approval of the eight `b` values:

```bash
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
QUESTION_MANIFEST=configs/experiments/activation_steering_preflight_8_20260725.jsonl \
EXPERIMENT_RUN_ID=activation_steering_qwen_csqa_tiny_20260725_v1 \
LAYERS=17,18 CONTROL_SEEDS=0,1 \
sbatch jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/tiny_dry_run_qwen_csqa.sbatch
```

The tiny job tests treatment batch size eight (falling back to one if the BF16
replay gate fails) and writes `compute_projection.json`. The full bundle has
separate pooled and dataset-specific direction fits, 116 layer-screen tasks,
12 selected-layer development dose tasks, 12 held-out tasks, 12 fixed-probe
tasks, four cross-dataset transfer tasks, and separate geometry/Alpaca stages.
The terminal CPU aggregation writes the behavioral dose/Pareto outputs plus
`supplementary_aggregate/` tables for fixed-probe ranks and margins, Alpaca
paired NLL changes, geometry summaries, and raw-versus-centered pair metrics.

After all evidence has passed review, print the exact approval template. The
inspection and tiny run directories retain immutable snapshots of the question
manifests used to produce them.

```bash
python3 scripts/validate_activation_steering_full_gate.py \
  --repo-dir "$PWD" \
  --config configs/experiments/activation_steering_controlled_20260725.json \
  --question-manifest configs/experiments/activation_steering_audited_1000_20260725.jsonl \
  --alpaca-manifest jobs/sycophancy_pruning/paper_global_sharded_20260722/evaluation/alpaca_utility.jsonl \
  --inspection-report /path/to/inspection/manifest.json \
  --tiny-compute-report /path/to/tiny/compute_projection.json \
  --expected-git-commit "$(git rev-parse HEAD)" \
  --print-approval-template
```

The researcher must review that template, fill the reviewer identity and
timezone-aware timestamp, change every explicit review assertion to `true`,
set `status` to `approved`, and enter the exact phrase
`APPROVE_CONTROLLED_ACTIVATION_STEERING_FULL`. Store the resulting JSON outside
the clean repository worktree and pass its path below.

## Structural validation

```bash
DRY_RUN=1 \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
EXPERIMENT_RUN_ID=activation_steering_controlled_preflight_20260725 \
bash jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/submit_activation_steering_controlled_sharded_20260725.sh
```

## DO NOT RUN YET — full submissions

```bash
TASK_FILTER=llama31_8b \
ALLOW_FULL_SUBMISSION=1 \
ACTIVATION_STEERING_INSPECTION_REPORT=/path/to/approved/inspection/manifest.json \
ACTIVATION_STEERING_TINY_COMPUTE_REPORT=/path/to/reviewed/tiny/compute_projection.json \
ACTIVATION_STEERING_FULL_GATE_APPROVAL=/path/to/reviewed/full_gate_approval.json \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
QUESTION_MANIFEST=configs/experiments/activation_steering_audited_1000_20260725.jsonl \
EXPERIMENT_RUN_ID=activation_steering_controlled_llama_20260725_v1 \
bash jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/submit_activation_steering_controlled_sharded_20260725.sh
```

```bash
TASK_FILTER=qwen25_7b \
ALLOW_FULL_SUBMISSION=1 \
ACTIVATION_STEERING_INSPECTION_REPORT=/path/to/approved/inspection/manifest.json \
ACTIVATION_STEERING_TINY_COMPUTE_REPORT=/path/to/reviewed/tiny/compute_projection.json \
ACTIVATION_STEERING_FULL_GATE_APPROVAL=/path/to/reviewed/full_gate_approval.json \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
QUESTION_MANIFEST=configs/experiments/activation_steering_audited_1000_20260725.jsonl \
EXPERIMENT_RUN_ID=activation_steering_controlled_qwen_20260725_v1 \
bash jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/submit_activation_steering_controlled_sharded_20260725.sh
```

All raw Slurm output is under `logs/<bundle>/slurm/<stage>/`. Canonical browseable
logs are written to `logs/<bundle>/by_task/<dataset_model>/<stage>/job_<id>/`.
