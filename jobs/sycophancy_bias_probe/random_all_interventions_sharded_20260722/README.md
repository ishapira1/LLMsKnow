# Pressure-restoration interventions stratified by `random_all` (2026-07-22)

This bundle is a prespecified causal follow-up to the strongest readout, `probe_bias_random_all`. It tests causal pressure restoration without treating post-answer decodability as if it were already a generation-time mechanism.

## What is confirmatory

The source probe uses the final token of a teacher-forced candidate answer. Its coefficient vector is therefore not automatically a valid pre-answer steering direction. The confirmatory interventions act at the final generation-prompt token and are:

1. Same-question neutral-to-strong-biased exact residual replacement at every nonterminal layer.
2. The reverse strong-biased-to-neutral patch.
3. A pre-answer pressure-restoration MeanDiff, `mean(h_neutral - h_strong_bias)`, fit on **train only** and balanced equally across `(correct option, user-backed option)` position strata.
4. Wrong-question donor, sign-reversed, matched-isotropic, and label-sign-null controls.

The current `random_all` direction is still used in two ways:

- It freezes the important test subgroup, `hidden_truth_flip`: the external answer follows the wrong user while `random_all` ranks the correct candidate first.
- Its direct pre-answer steering and probe-parallel/orthogonal patch decomposition are run as explicitly labeled **exploratory cross-token-position analyses**. A failure there does not falsify the post-answer probe.

## Split discipline

The stages are deliberately separated:

1. `fit_directions_array.sbatch`: fit MeanDiff on train only, one model/dataset cell per job.
2. `localize_layers_array.sbatch`: sweep every nonterminal residual layer (31 for Llama-3.1-8B, 27 for Qwen2.5-7B) using only paired patches and patch controls on validation. The final-normalized state is excluded because replacing it with the neutral state would deterministically replace the lm-head input and give a tautological result.
3. `select_layers_array.sbatch`: freeze the top three validation layers that show bidirectional recovery and beat wrong-question and matched-random patch controls.
4. `dose_tune_array.sbatch`: run the signed MeanDiff grid and five seeded null/random directions at those candidate layers on validation. Controls run on neutral, wrong-suggestion, and correct-suggestion prompts too.
5. `select_dose_array.sbatch`: select layer × dose using the within-item difference-in-differences `Δmargin(strong wrong suggestion) − Δmargin(neutral)`, require it to beat both controls, require the matched negative dose to move oppositely and a layer-level dose-response Spearman correlation of at least 0.70, and apply neutral/genuine-agreement guards. `hidden_truth_flip` items are explicitly excluded from both validation-selection stages.
6. `confirm_test_array.sbatch`: evaluate the frozen layer and `{0, +alpha, -alpha}` on the full test split, with 20 seeded null/random directions and matched-random patches. Capped training pilots are forbidden here.
7. `aggregate_array.sbatch`: aggregate one shard at a time, create exhaustive question-level normal intervals, bootstrap only the frozen primary estimands/contrasts, and estimate the `random_all` treatment-moderator interaction.

All evaluated conditions are replayed before any intervention. A shard stops if any condition has less than 98% top-choice agreement or if its 99th-percentile absolute choice-probability error exceeds 0.01, because the June source runs did not pin an exact Hugging Face checkpoint revision. Replay-matched subsets are also frozen for the primary report.

## Primary estimands

For correct choice `c` and user-backed wrong choice `b`, the primary continuous outcome is:

`Δmargin = Δ [log P(c) - log P(b)]`, computed directly from stable choice log-scores.

The steering selection and mitigation estimand is:

`Δmargin(strong wrong suggestion) - Δmargin(neutral)`.

Also reported are `P(c)`, `P(b)`, accuracy, targeted endorsement, condition-specific suggestion agreement/probability, reversals, induced endorsed errors, entropy, KL/TV distribution shift, and normalized recovery toward the paired neutral margin. Summaries include question-clustered bootstrap intervals for all items and for the frozen subgroups `neutral_correct`, `sycophantic_flip`, `hidden_truth_flip`, `sycophantic_flip_probe_user`, `sycophantic_flip_probe_other`, and `neutral_wrong_to_correct_suggestion_correct`. Multiple random-control seeds are averaged within question before questions are bootstrapped.

The central success pattern is:

- neutral-to-biased full patching raises the `c` versus `b` margin;
- the reverse patch lowers it;
- positive MeanDiff has a signed dose response and beats null/random controls;
- its wrong-pressure-minus-neutral effect is positive, so the result is not merely a generic shift toward one answer;
- the frozen test intervention recovers hidden-truth flips;
- recovery is larger when `random_all` ranks truth than when it follows the user, establishing the probe as a treatment moderator (not yet a pre-answer causal truth axis);
- neutral and correct-suggestion accuracy each fall by no more than 2 percentage points.

## Run

First validate shell syntax and the submission graph:

```bash
for file in jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/*.sbatch jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/*.sh; do bash -n "$file"; done
DRY_RUN=1 bash jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/submit_random_all_interventions_sharded_20260722.sh
```

Then submit:

```bash
bash jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/submit_random_all_interventions_sharded_20260722.sh
```

Useful validation-pilot overrides are `TASK_FILTER`, `FIT_MAX_QUESTIONS`, and `VAL_MAX_QUESTIONS`. Empty max-question values mean the full split. A capped direction artifact is blocked from held-out confirmation, and there is deliberately no `TEST_MAX_QUESTIONS` option. Each submission gets a unique `EXPERIMENT_RUN_ID` output namespace, and writers refuse to overwrite an existing shard. `ALLOW_STALE_LOCK_CLEANUP` is intentionally not used; this bundle never removes source locks or mutates source probe runs.

## Outputs

Source June probe artifacts remain immutable. New artifacts live under:

```text
.../LLMsKnow_results/sycophancy_bias_intervention/random_all_interventions_20260722/<run_id>/
  <dataset_model>/
    directions/{directions.npz,manifest.json,pair_coverage.csv}
    layers/layer_NNN/{val,test}/item_results_{patch_localize,dose_tune,confirm}.jsonl
    selected_patch_layers.json
    selected_intervention.json
    validation_layer_selection.csv
    validation_dose_selection.csv
    aggregate/{item_result_catalog,summary,primary_bootstrap,causal_contrasts,probe_moderator}_all_splits.csv
```

Logs follow the repository convention under:

```text
.../LLMsKnow_logs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/
  submit/
  slurm/{fit,localize,select_layers,dose_tune,select_dose,confirm,aggregate}/
  by_task/<dataset_model>/<stage>/job_<job_id>/task_<array_task>.{out,err}
```
