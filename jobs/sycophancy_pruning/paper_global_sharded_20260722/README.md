# Paper-faithful sycophancy pruning

This bundle implements the primary experiment with signed, dataset-average
pruning attribution and one absolute, dataset-average mixed-preservation
attribution. Selection is global across every `nn.Linear.weight` inside a
transformer block:

```text
bottom_q(prune_score) \ top_p(preservation_score)
```

There is no per-pool, per-layer, or per-matrix normalization. The bundle uses
model-specific manifests and masks for Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct.

## Locked primary flags

Score jobs use:

```text
--prune_method attribution_score_set_difference_global
--score_format raw
--loss_mode completion_nll
--attribution_variant paper
--no_abs
--neg_prune
--abs_preserve
--dump_score
```

Mask jobs replace `--dump_score` with:

```text
--use_saved_scores
--p <p>
--q <q>
--dump_mask
```

`q=0` is handled as a direct, unmodified base checkpoint and does not load a
score cache. The omitted flags remain false, and `alpha=0` remains the pruning
default.

The full grid is:

```text
p = 0, 1e-5, 5e-5, 7e-5, 1e-4
q = 1e-6, 3e-6, 7e-6, 1e-5, 2e-5, 5e-5, 1e-4
```

## Before submitting

Both the model and tokenizer must be pinned to the same immutable Hugging Face
commit SHA. Export the revisions in the submission shell so Slurm propagates
them to every job:

```bash
export QWEN_REVISION=<qwen_commit_sha>
export LLAMA_REVISION=<llama_commit_sha>
export ALPACA_DATA=/path/to/alpaca_data_cleaned_archive.json
```

Defaults assume these cluster checkouts and interpreter:

```text
REPO_DIR=/n/home12/ishapira/LLMsKnow
HARM_REPO_DIR=/n/home12/ishapira/harm_pruning_WIP
ENV_PYTHON=/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python
```

Override them before submission if the checkouts or environment live
elsewhere. `.env` must configure a Hugging Face cache outside `/home`; gated
Llama access must already be available.

Always validate a stage before submitting it:

```bash
bash -n jobs/sycophancy_pruning/paper_global_sharded_20260722/*.sh \
  jobs/sycophancy_pruning/paper_global_sharded_20260722/*.sbatch

DRY_RUN=1 \
  jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh setup
```

## Run order

Run setup once. It samples actual generated answer identities for both models,
both datasets, and calibration seeds `5`, `17`, and `29`; builds the strict
nested manifests; builds one fixed seed-5 held-out val/test manifest; and
snapshots raw/chat tokenization.

The held-out manifest also freezes two semantic rephrasings of every weak
wrong suggestion. The unmodified `q=0` checkpoint and every selected mask are
run live on those exact prompts; cached weak-prompt responses are retained only
as provenance and are never substituted for paraphrase results.

```bash
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh setup
```

Wait for setup to finish. A main manifest fails instead of shrinking if either
dataset cannot supply 206 strict flips for a model/seed.

Then run the nested calibration tiers in order, inspecting each selection
before increasing scale:

```bash
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh smoke
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh pilot
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh main
```

Each tier computes the seed-5 primary scores once, evaluates the unmodified
base model, runs sequential `q` waves over all five `p` values, and selects on
validation. Later waves skip a `(model,p)` branch after a predeclared hard
utility stop. The stop uses the declared utility limits: 2-point drops in
neutral accuracy/probability, corrective resistance, or correct-suggestion
agreement; 5% preservation-loss or perplexity increases; and a 2-point rise in
other-wrong/invalid outputs. They can be overridden with
`HARD_NEUTRAL_ACCURACY_DROP`, `HARD_NEUTRAL_PROBABILITY_DROP`,
`HARD_CORRECTION_ACCURACY_DROP`, `HARD_AGREEMENT_ACCURACY_DROP`,
`HARD_PRESERVATION_LOSS_INCREASE`, `HARD_WIKITEXT_PPL_INCREASE`, and
`HARD_OTHER_WRONG_INVALID_INCREASE`.

After the main seed-5 selection completes:

```bash
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh controls
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh replications
```

`controls` runs structure-matched, opposite-sign, second-slice,
magnitude-matched random, Alpaca-only preservation, chat, choice-token, and
released-absolute comparisons at the selected seed-5 `(p,q)`. `replications`
holds `(p,q)` fixed and recomputes primary masks for calibration seeds `17` and
`29`.

Only after controls and replications have finished, run the test cohort once:

```bash
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh final-test
```

`final-test` automatically submits one CPU-only `report` job after the complete
base/seed-5/seed-17/seed-29 test array succeeds. The report builder reads the
canonical experiment root and writes the minimum result package to:

```text
<experiment-root>/reports/minimum_result_package/
```

That package is intended to contain the transition chart, truth-restoration
bars, preservation deltas, intervention comparison, sparsity tradeoff, and
generalization heatmap. Preservation deltas include all three calibration
masks, and every control is checked against its recorded sign/slice/control
metadata before plotting. Loss and perplexity values are read from immutable,
hashed evaluation snapshots saved beside each offline comparison. To rebuild it
without rerunning model evaluation:

```bash
jobs/sycophancy_pruning/paper_global_sharded_20260722/submit_paper_global_sharded_20260722.sh report
```

Use `REPORT_BOOTSTRAP_SEED` to change the report bootstrap seed, or
`REPORT_OUTPUT_DIR` to retain an additional report build. The builder refuses a
nonempty output directory by default so stale figures cannot survive a rebuild;
set `REPORT_OVERWRITE=1` only when intentionally replacing that exact report
directory. To chain a standalone report behind an externally submitted final
job, also set `DEPENDENCY_JOB_ID=<job_id>`.

Final test enables zero-shot utility and the Alpaca benign-instruction judge
for the base and the three primary calibration masks, and saves each selected
masked checkpoint with its tokenizer and metadata. The judge uses the fixed
`evaluation/alpaca_utility.jsonl` artifact generated during setup. That cohort
is deterministic and disjoint from every seed's mixed and Alpaca-only
preservation rows; `ALPACA_DATA` is only its source dataset. Set `RUN_ZERO_SHOT=0`,
`RUN_ALPACA_EVAL=0`, or
`SAVE_SELECTED_MODEL=0` only for an explicitly reduced run. Alpaca judging
requires `COHERE_API_KEY` (or `COHERE_KEY`) and can be costly; control its size
with `ALPACA_EVAL_NSAMPLES`. Its subset seed is independent of the calibration
seed and fixed at `ALPACA_EVAL_SEED=5`. Final selected jobs default to serial execution to
avoid judge rate-limit bursts; override `SELECTED_ARRAY_CONCURRENCY` only when
the service quota permits it. WikiText perplexity and the matched
preservation-manifest loss always run. Final paired question-clustered
bootstrap intervals use 2,000 resamples by default.

If seed-5 validation has no feasible mask, the selection artifact explicitly
records `no_feasible_mask`; downstream selected-mask jobs retain the base model
and skip mask controls/replications.

## What each job does

- `sampling_array.sbatch`: actual deterministic MC generation plus audit choice
  probabilities; sample batch size defaults to one.
- `manifests_array.sbatch`: strict flip filtering, 16/128/412 nested sets, the
  40/30/15/15 mixed preservation set, a disjoint structure control, Alpaca-only
  control, overlap audits, hashes, and the fixed held-out manifest with two
  frozen weak-suggestion paraphrases per question.
- `token_snapshot_array.sbatch`: raw/chat renderings, token IDs, and explicit
  response spans for pruning rows and the complete smoke preservation mix
  (including Alpaca) for both pinned tokenizers.
- `score_array.sbatch`: independently sharded prune/preserve score jobs.
- `mask_eval_grid_array.sbatch`: exact global masks, preservation loss,
  WikiText perplexity, live held-out inference, offline metrics, and hard-stop
  sentinels.
- `select_array.sbatch`: exact feasibility filtering and the declared
  recovery-first tie break.
- `selected_mask_eval_array.sbatch`: selected controls, independent-seed masks,
  and paired clustered bootstrap evaluation.
- `mask_overlap_array.sbatch`: exact overall and per-module overlap for the
  selected seed-5, seed-17, and seed-29 primary masks.
- `report.sbatch`: CPU-only assembly of the minimum result package from the
  completed experiment root; it runs automatically after `final-test`.

## Artifacts and logs

Experiment artifacts live beneath:

```text
<cache-parent>/LLMsKnow_results/sycophancy_pruning/paper_global_sharded_20260722/
```

The tree separates `sampling/`, `manifests/`, `pruning_artifacts/`,
`predictions/`, `analysis/`, `reports/`, and identity-resolved `registry/` pointers. Score
and mask paths include model revision, manifest hashes, format, loss, seed,
sample counts, `p`, `q`, and sign/control settings. Sparse masks include the
exact eligible parameter universe, nominal counts, surviving counts, and
per-module counts.

Grid, prediction, registry, hard-stop, and selection paths also contain a
content identity of the form
`prune_<hash>_preserve_<hash>_eval_<hash>`. This identity is recomputed from
the exact pruning, preservation, and fixed held-out manifests before every
stage. Regenerating a manifest therefore starts a separate grid and cannot
reuse an earlier hard-stop sentinel, validation summary, or selected
configuration. Control and replication paths use the corresponding
seed/variant manifest identity; mask-overlap output uses a combined identity
for all three calibration manifests.

Logs are under:

```text
jobs/sycophancy_pruning/logs/paper_global_sharded_20260722/
```

Use `submit/` for submission records, `slurm/<stage>/` for raw scheduler logs,
and `by_task/<model>/<stage>/job_<id>/task_<array>.{out,err}` for canonical task
logs. Stale `.run.lock` files are never removed unless
`ALLOW_STALE_LOCK_CLEANUP=1` is explicitly set.

The `replications` stage automatically writes exact pairwise mask overlap. To
rerun the comparison manually after the three primary masks exist:

```bash
PYTHONPATH=src "$ENV_PYTHON" scripts/compare_pruning_masks.py \
  --mask 5=/path/to/seed5/indices.pt \
  --mask 17=/path/to/seed17/indices.pt \
  --mask 29=/path/to/seed29/indices.pt \
  --output /path/to/mask_overlap.json
```

The manifest bundle already reports question-ID and suggested-label overlap
across seeds.
