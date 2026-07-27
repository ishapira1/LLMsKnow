# Conditioned activation-steering gate (2026-07-26)

This bundle implements the preregistered saved-activation audit and the
subsequent bounded ARC causal gate. Stages are intentionally submitted one at
a time. A later stage must not be submitted until the preceding immutable
decision artifact authorizes it.

The run identity is derived from the configuration and question-manifest
hashes. All output writers refuse overwrite.

## Gate order

1. `stage_a_audit.sbatch`: CPU-only mean-cancellation audit. Continue only
   when `stage_a_audit/decision.json` has `gpu_stage_authorized=true`.
2. `build_cohorts.sbatch`: freeze model-specific ARC validation/test cohorts
   whose saved neutral top-1 is correct.
3. `bf16_gate_array.sbatch`: real-model BF16, rendered-offset, nonfinite, and
   exact same-shard alpha-zero gate for both models.
4. `project_compute.sbatch`: benchmark-derived cost projection and mandatory
   reduction sequence; continue only below 48 accelerator-hours.
5. `validation_array.sbatch`: screen three nominated layers, two position
   modes, comparators, and the full ratio grid.
6. `select_validation.sbatch`: apply the preregistered behavioral selection
   rule. Models without an eligible candidate stop here. A zero-candidate
   result is a successful scientific stop, not an operational failure.
7. `finalize_validation_stop.sbatch`: when no model is selected, materialize
   the immutable negative decision, candidate/model tables, and validation
   plot. Do not submit any held-out GPU stage in this branch.
8. `test_learned_array.sbatch`, `test_controls_array.sbatch`, and
   `suffix_sensitivity_array.sbatch`: held-out learned curves, 20 item-sign and
   20 isotropic controls, and the capped same-per-position sensitivity.
9. `aggregate_test.sbatch`: paired intervals, null comparison, position test,
   plot, and machine-readable final decision.

Use the stage submitter with the same environment variables used for Stage A:

```bash
DRY_RUN=1 STAGE=cohort \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_conditioned_gate_20260726.json \
QUESTION_MANIFEST=configs/experiments/activation_steering_signal_300_20260726.jsonl \
EXPERIMENT_RUN_ID=activation_steering_conditioned_gate_20260726_v2 \
bash jobs/sycophancy_bias_probe/activation_steering_conditioned_gate_sharded_20260726/submit_stage_b_gate_20260726.sh
```

Replace `cohort` with exactly one subsequent stage name after inspecting its
gate artifact: `bf16`, `projection`, `validation`, `selection`,
`finalize-stop`, `test`, `controls`, `sensitivity`, or `aggregate`. Use
`finalize-stop` only when selection saved zero eligible models; use the
held-out stages only for selected models.

No script in this bundle automatically submits a broader DAG.
