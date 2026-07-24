# Independent Qwen MeanDiff validation pilot (2026-07-24)

This bundle is an overnight, validation-only pilot designed to produce an interpretable
MeanDiff dose-response result while the full CSQA intervention DAGs remain queued.
It does not modify, depend on, reuse outputs from, or unlock test confirmation for those DAGs.

The fixed direction is the full-train Qwen direction from
`random_all_csqa_qwen_full_20260723_v4`. The pilot evaluates 160 validation questions at
six prespecified nonterminal layers (`3,8,13,18,23,27`), the full signed alpha grid, and
three train-derived null/random control seeds. The transported post-answer probe analysis is
disabled because it is outside the overnight question and adds compute.

The six GPU shards request typed H200 GPUs from `gpu_requeue`. They may be preempted and
requeued. A dependent CPU job applies the existing dose-selection gates, aggregates the
question-level results, and creates a multi-layer dose-response CSV and plot. If no layer/dose
passes every gate, that no-go is retained as a scientifically meaningful pilot outcome and
aggregation still runs.

Run syntax and dry-run checks:

```bash
bash -n jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/*.sbatch \
  jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/*.sh

DRY_RUN=1 \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
bash jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/submit_qwen_meandiff_pilot_20260724.sh
```

Submit:

```bash
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
bash jobs/sycophancy_bias_probe/qwen_meandiff_pilot_20260724/submit_qwen_meandiff_pilot_20260724.sh
```

The immutable submission manifest contains both job IDs and the exact output/log roots.
