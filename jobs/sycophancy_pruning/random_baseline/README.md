# `random_baseline` Slurm campaign

This bundle implements the preregistered Llama Mixed-996 and Qwen 3,139-weight
specificity controls. It builds 20 `uniform_global` and 20
`module_magnitude_matched` masks per model, audits them, evaluates the fixed
base-neutral-correct cohorts, runs five fixed seeds through the common suite,
packages blinded feedback judgments, and produces verified lightweight report
assets. No raw generations or masks are committed.

## Frozen design

- Seeds: `101, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201,
  1301, 1409, 1511, 1601, 1709, 1801, 1901, 2003`.
- Broad seeds: `101, 503, 1009, 1511, 2003`.
- Confirmatory control: exact per-matrix counts and exact 20-bin absolute-weight
  magnitude counts, disjoint from the learned mask.
- Primary endpoint: strong wrong-suggestion adoption on the base-neutral-correct
  cohort. Guardrails: neutral accuracy and invalid-answer rate.
- Equivalence: 3 percentage points on adoption and 2 points on neutral accuracy.
- Confirmatory rank: `(1 + random effects at least as strong as learned) / 21`.
  Cross-model support requires `1/21`, no equivalents, and the neutral guardrail
  for both models.

The human-readable design and machine-readable preregistration live at
`artifacts/pruning/random_baseline/`.

The primary Qwen cohort remains all 256 frozen questions. Its paraphrase audit
uses every final-cohort question with a pre-existing valid frozen stem and does
not replace or reselect missing questions; exact coverage and exclusions are
hash-pinned by preflight.

## Efficiency and integrity

The uniform sampler uses small-`K` rejection sampling over cumulative module
offsets. The matched builder computes each target matrix's magnitude assignment
once, reuses it across all seeds, and samples only the required coordinates per
bin. Each evaluation shard loads its model once, backs up the union of affected
coordinates, then restores and verifies that union between states. Batch size 1
is retained unless a byte-identical batch-8 path is certified; the current
parity gate conservatively records batch 1.

`preflight_pins.json` is immutable after creation. Every reuse requires exact
prompt, model revision, mask, decoding, and artifact hashes. Existing partial
directories are fatal. `.run.lock` cleanup is never automatic;
`ALLOW_STALE_LOCK_CLEANUP=1` is required for an explicitly reviewed recovery.

## Launch on Cannon

Authenticate the reusable SSH master first from the local machine:

```bash
ssh cannon
```

In the clean cluster worktree, run all static checks and the required dry run:

```bash
bash -n jobs/sycophancy_pruning/random_baseline/*.sh
bash -n jobs/sycophancy_pruning/random_baseline/*.sbatch
PHASE=validation DRY_RUN=1 jobs/sycophancy_pruning/random_baseline/submit_random_baseline.sh
PHASE=validation DRY_RUN=0 jobs/sycophancy_pruning/random_baseline/submit_random_baseline.sh
```

Wait for the reported validation job to complete and verify both smoke/timing and
batch-decision artifacts. Then separately dry-run and submit the campaign:

```bash
PHASE=campaign DRY_RUN=1 jobs/sycophancy_pruning/random_baseline/submit_random_baseline.sh
PHASE=campaign DRY_RUN=0 jobs/sycophancy_pruning/random_baseline/submit_random_baseline.sh
```

The campaign phase ends after packaging. Arrays are capped at four simultaneous GPUs.
Raw Slurm logs are submitted under `slurm/<stage>/`; canonical logs are copied
live to `by_task/<model>/<stage>/job_<id>/task_<array>.{out,err}`. Each log has
the task identity, command, model, revision, paths, Slurm IDs, host, working
directory, start/end times, status, elapsed seconds, and resource snapshots.

## Local blinded judging and recovery

Download `judging/feedback_packet.jsonl` only (not the private key), then run:

```bash
python jobs/sycophancy_pruning/random_baseline/judge_feedback_with_codex.py \
  --packet feedback_packet.jsonl --output feedback_labels.jsonl
```

The judge is resumable and writes atomically after every completed chunk. Upload
the labels to `$RESULT_ROOT/judging/feedback_labels.jsonl`, then dry-run and
submit final aggregation:

```bash
DRY_RUN=1 jobs/sycophancy_pruning/random_baseline/resume_after_judging.sh
DRY_RUN=0 jobs/sycophancy_pruning/random_baseline/resume_after_judging.sh
```

For a terminally failed Slurm task, use the narrow audited recovery entry point;
it requires an exact stage, array task/range, and unique ID, and defaults to a
dry run. It never removes locks unless `ALLOW_STALE_LOCK_CLEANUP=1` is supplied:

```bash
DRY_RUN=1 RECOVERY_STAGE=core RECOVERY_ARRAY=7 RECOVERY_ID=core_task7_r1 \
  bash jobs/sycophancy_pruning/random_baseline/recover_stage.sh
```

## Acceptance and release

Do not release until `audit/completion_audit.json` is complete; all 80 masks,
all core states, all 12 broad states per model and benchmark, feedback labels,
automatic ELEPHANT labels, milestone receipts, and the final inference must be
present. `export_report.py` then writes only lightweight artifacts and a generated
`random_mask_baselines.tex`. Insert that subsection immediately before “What
Weights Made the Difference?” in `hidden_sycophancy/experiemnts.tex`, copy the
two Pareto PDFs to `plots/`, compile with missing-reference warnings treated as
fatal, and only then integrate the campaign commit into the latest `main` of
each repository. The final progress email must include both pushed SHAs, the
Harvard result path, headline statistics, and the completion-audit hash.

## User-authorized core-complete early stop

If the user explicitly authorizes stopping after the complete confirmatory core,
use `early_stop.py`; do not relabel the original full-suite audit as complete.
The early-stop path requires all 80 audited controls, all 84 core states, the
exact two-model confirmatory inference, and no scheduler work left queued. Its
resource-saving rule requires every random seed to remain within 2 percentage
points of base strong-wrong adoption and within 1 point of base neutral
accuracy, with unchanged invalid-answer rates and at least 95% answer
invariance. Any already-finished broad block is hash- and row-audited, while
omitted broad and judging work is recorded explicitly.

After holding pending work and completing a dry-run target inspection, cancel
only the reviewed job IDs. Then run:

```bash
python jobs/sycophancy_pruning/random_baseline/early_stop.py finalize \
  --result-root "$RESULT_ROOT" \
  --job-ids 36927495 36927496 36927497
```

Send the idempotent `early_stop_decision` and `final_report_complete`
milestone emails, then create the immutable alternative audit:

```bash
python jobs/sycophancy_pruning/random_baseline/early_stop.py audit \
  --result-root "$RESULT_ROOT" \
  --job-ids 36927495 36927496 36927497
```

The resulting `audit/early_stop_completion_audit.json` authorizes only a
core-complete early-stop export. It never implies that feedback judging or all
144 broad states ran.
