# Cross-model confidence–resistance bundle (2026-08-28)

This bundle runs the frozen ARC-Challenge/CommonsenseQA experiment implemented
in `src/llmssycoph/confidence_resistance.py`.

The dependency graph is:

1. Freeze 200 discovery, 600 confirmation, and 400 reserve questions per dataset.
2. In parallel, run the GPT equal-bias pilot/full pipeline and the four
   Llama/Qwen neutral-scoring tasks.
3. Freeze model-specific neutral-rank 2, rank 3, and last targets.
4. Run the four Llama/Qwen endorsement arrays.
5. Audit coverage and analyze discovery data only.

Confirmation is deliberately not submitted automatically. After reviewing the
discovery diagnostics and verifying that `frozen_analysis_spec.json` has not
changed, submit `confirmation_analysis.sbatch` with
`UNLOCK_CONFIRMATION=1`.

If the initial QC audit has any confirmation cell below 400, run the reserve
submitter. It first verifies that the audit failed, then scores the pre-frozen
reserve cohort and promotes only a common, QC-complete deterministic prefix:

```bash
PRICING_RECHECKED=1 OPENAI_CONFIRM_SPEND=1 DRY_RUN=1 \
  jobs/sycophancy_bias_probe/confidence_resistance_20260828/submit_reserve.sh
```

Change `DRY_RUN` to `0` only after inspecting the reserve request counts.

Run a dry run first:

```bash
DRY_RUN=1 jobs/sycophancy_bias_probe/confidence_resistance_20260828/submit.sh
```

Production GPT submission requires a fresh official pricing check and explicit
acknowledgement:

```bash
PRICING_RECHECKED=1 OPENAI_CONFIRM_SPEND=1 DRY_RUN=0 \
  jobs/sycophancy_bias_probe/confidence_resistance_20260828/submit.sh
```

The GPT pilot uses unbiased and equally biased requests. If the pilot fails,
the pipeline switches to unbiased top-20 scoring, retains missing letters as
censored, and disables the universal point-estimate conclusion.

Raw results default to Holystore. Canonical task logs are written under
`by_task/<dataset_model>/<stage>/job_<job_id>/task_<array_task>.out/.err`.
No script removes `.run.lock` files.
