# Controlled Prompt-Only Activation Steering Preflight

**Protocol:** `controlled_prompt_only_v1_20260725`

**Status:** implemented, CPU contract-tested, blocked on human semantic review and the first GPU inspection
**Full submission:** **DO NOT RUN YET**

## Scientific question

The experiment tests whether a shared prompt-only activation displacement caused
by a false user suggestion selectively controls agreement with the user-backed
wrong answer. It separates an external policy effect from changes in the fixed
`random_all` candidate-answer readout and from nonselective model damage.

For question \(i\), ordinary wrong pressure \(W\), and neutral framing \(N\):

\[
\delta_i^{WN}=h_i^W-h_i^N,\qquad
v_{WN}=\frac{1}{n}\sum_i\delta_i^{WN},\qquad
h'=h+\alpha v_{WN}.
\]

Positive alpha means more like ordinary wrong pressure. Alpha one adds exactly
one raw paired mean shift. Direction fitting is model- and layer-specific,
uses training questions only, and never contains assistant answer tokens.

## Canonical sources read

- `sources/README.md`: canonical-results policy.
- `sources/00_project_brief (3).md`: research question and causal gaps.
- `sources/current_empirical_summary_2026-06-20.md`: canonical empirical
  behavior, confidence, probe, transfer, and movement findings.
- `../meeting_record_2026-07-24.md` and
  `sources/meeting_notes_2026-07-24.md`: preliminary steering observation and
  current priorities.
- `sources/three_claims_probe_framework_project_note (3).md`: supported claim
  ladder and limits of probe evidence.
- `sources/hidden_factual_methodology_for_sycophancy (4).md`: candidate-answer
  feature and leakage methodology.
- `AGENTS.md`: plot and Slurm requirements.

Implementation audit covered:

- `src/llmssycoph/interventions/{data,activations,directions,experiment,metrics,plots}.py`
- `src/llmssycoph/probes/features.py`
- `src/llmssycoph/data/prepare.py`
- `tests/test_random_all_intervention_contract.py`
- `jobs/sycophancy_bias_probe/random_all_interventions_sharded_20260722/`
- pulled direction manifests, Qwen pilot manifests/rows, selected intervention,
  frozen random-all probes, and retry logs under `cluster_pull_20260724/`.

The June 20 summary is the only source used for existing quantitative empirical
claims. July 24 notes and old steering outputs are planning/pilot evidence only.

## What the legacy implementation actually does

The legacy code fits

\[
d=\operatorname{balancedMean}_{(c,b)}(h^N-h^{S}),
\quad
h'=h+\alpha\,\sigma_{\mathrm{proj}}\,d/\|d\|.
\]

It therefore differs from the controlled replication in four ways:

1. it uses strong wrong pressure rather than ordinary wrong pressure;
2. it points neutral-minus-pressure, so positive alpha removes pressure;
3. it balances option strata rather than taking an unweighted question mean;
4. alpha is measured in projection-standard-deviation units, not raw mean shifts.

The legacy path is preserved under the interpretation
`legacy_restoration_v0`. Its artifacts and alphas must not be pooled with the
controlled protocol.

## Exact activation path

```text
saved sampling row
→ build_intervention_pairs(require_metric_usable=False)
→ saved prompt_messages
→ generation._resolve_model_inputs()
→ human role mapped to user
→ tokenizer.apply_chat_template(add_generation_prompt=True)
→ forward(use_cache=False, output_hidden_states=True)
→ hidden_states[layer][:, -1, :]
→ post-block residual hook at the final rendered prompt token
→ raw alpha-scaled addition
→ option-label token log-sum-exp scores
→ normalized A–E probabilities and explicit question row
```

The token at `-1` is the model-native assistant-start token inserted by the chat
template, not the colon in the literal `Answer:` text. Nonterminal hidden-state
index `l` is the output of decoder block `l-1`, before the next block's
pre-normalization. The hook sees `[batch, sequence, hidden]`, changes only the
audited prompt-boundary position, and is removed in `finally`.

For fixed-probe scoring, each answer candidate is teacher-forced. Steering is
applied at the exact saved prompt boundary, while the frozen `random_all`
classifier reads the last candidate-answer token at its original feature layer.
If steering is at or downstream of the probe layer, probe invariance is labeled
structurally uninformative.

## Critical audit findings

### Alpha-zero failure

The Qwen validation pilot is not an exact no-op. Inspected layer manifests report
maximum zero-path total variation `0.05188038347902866`; item rows reach a
`1.375` absolute c-versus-b margin shift despite no top-choice flips. Baselines
were batch-one, while the zero/intervention path used expanded BF16 batches.
The pilot is therefore unconfirmed and supplies no causal result.

The controlled path now:

- returns the original hook output object for a zero vector;
- compares disabled and zero-hook paths at the identical batch shape;
- requires bitwise-identical same-shape probabilities and margins;
- gates cross-batch probability error at `0.005`, c-b margin error at `0.05`,
  and top-choice agreement at 100%;
- automatically falls back to batch size one after a cross-batch failure and
  records that decision in the compute report.

### Response-conditioned fitting

The old pairing contract required generated responses to be usable for metrics.
The controlled direction path checks prompt/question structure only. It may
compute baseline subgroup labels for later reporting, but never uses them for
direction membership.

### Semantic validity of `b`

Code only guaranteed `b != c`. A discovered CSQA item treated both “homely” and
the selected `b="ugly"` as plausible answers. The full run is now hard-gated on
a human-reviewed manifest. Candidate generation is deterministic, balanced by
dataset/split/endorsed label, and cannot use post-steering outcomes.

## Splits and manifests

Existing membership is preserved to avoid contaminating the fixed probes:

- CSQA seed 5: 7,016 train, 1,754 validation, 2,192 test.
- ARC source splits after duplicate collapse: 1,118 train, 299 validation,
  1,172 test.

The controlled stable identity is `(dataset, source_example_id)`; `q_<index>` is
display-only. The target audited cohort is 500 questions per dataset:
300 train, 100 validation, and 100 test. All framings stay with the question.
ARC source labels `1–5` are retained for model-facing token scoring but mapped
positionally to canonical `A–E` fields for balance checks, manifests, probe
ranks, and result tables. Both label systems and the exact mapping are saved.

`configs/experiments/activation_steering_preflight_8_20260725.jsonl` contains
three train, two validation, and three test examples. All eight were explicitly
approved by `itaishapira@g.harvard.edu` on 2026-07-25 after review of inspection
job `35149100`; the recorded decision depends only on semantic wrongness of
`b`. Direction fitting and interventions still refuse any manifest containing
a non-approved row or incomplete reviewer provenance.

The full audited manifest is intentionally absent. Its schema and quotas are in
`configs/experiments/activation_steering_audited_1000_20260725.schema.json`.
`scripts/build_activation_steering_audit_candidates.py` creates a two-times
review pool plus CSV without claiming that semantic review has happened.
The deterministic pending-review artifacts are
`configs/experiments/activation_steering_review_pool_2000_20260725.jsonl` and
`configs/experiments/activation_steering_review_pool_2000_20260725.csv`.
`scripts/freeze_activation_steering_audited_manifest.py` then takes approved
rows only, replenishes rejections within dataset/split/endorsed-label strata,
and freezes the exact 1,000-row manifest.

The selected utility dose is chosen on development data only, at the selected
layer. Among the predeclared symmetric screen magnitudes, it maximizes
`P(b|W,+m) - P(b|W,-m)` minus the mean neutral probability, neutral accuracy,
and neutral final-prompt-only invalid-output damage. All-condition generation
degeneration remains a separate gate. The choice must pass the same accuracy,
validity, and degeneration gates to remain confirmatory; otherwise it is
explicitly descriptive. Alpaca uses `−128`, the selected negative dose, `0`,
the selected positive dose, and `+128`.

## Directions and controls

All paired direction differences and means are computed and stored in float32:

- `WN = ordinary wrong − neutral` — primary replication.
- `CN = correct suggestion − neutral`.
- `WC = ordinary wrong − correct suggestion`.
- `SW = strong wrong − ordinary wrong`.

No answer-free generic-pressure family exists, so `GN` is unavailable.

For deterministic seeds 0–9:

- isotropic Gaussian vectors are matched to `||WN||`;
- coordinate-sign controls preserve every WN coordinate magnitude;
- balanced item-sign means are stored in native units and matched to `||WN||`;
- CN, WC, and SW are evaluated in native and WN-matched units.

Every artifact records direction/residual norms, median item-shift norm, sign
orientation, centroid movement, model/source/manifests, revisions, git/dirty
fingerprints, and exact commands. NaN and infinity are hard failures.

## Alpha, outputs, and metrics

The primary grid is:

```text
[-128,-64,-32,-16,-8,-4,-2,-1,-.5,-.25,0,
 .25,.5,1,2,4,8,16,32,64,128]
```

The tiny run must extend by powers of two if the endpoint does not reach an
injected-to-residual norm ratio of four. Extreme points are retained as
diagnostics.

Question-level output explicitly saves A–E option log scores and probabilities,
correctness, equality to `b`, error and targeted-error indicators, P(c), P(b),
log-score/probability margins, entropy, residual and injected norms, alpha
convention, the exact direction/control formula, prompt token count, selected
token index/ID/text, model-specific option token IDs, subset labels fixed before
steering, fixed-probe scores/ranks, and provenance. Aggregates include the
paired-bootstrap targeted-error share among errors. Strict option scoring is
distinguished from free-generation validity diagnostics.

Aggregation uses paired question bootstrap intervals. The primary plots are a
dose-response curve with 10-seed control ribbons and a sycophancy-reduction
versus neutral-damage Pareto plot. Plotting follows repository Seaborn styling
and places legends below plots. Whenever both datasets are present, tables
include an explicit pooled ARC+CSQA scope in addition to dataset-specific rows.
Held-out dose plots retain separate selected-layer/neighbor curves instead of
averaging the three intervention sites together.

The immutable shard JSONL files remain the explicit wide question-level table.
To keep the CPU aggregation stages bounded, aggregation retains learned
treatments and every alpha-zero replay row at question level, while reducing
nonzero stochastic controls to exact per-seed weighted means after each shard
is read. Paired question bootstrap intervals therefore apply to learned
treatments; compacted controls are explicitly marked as not bootstrapped and
their declared uncertainty is the across-seed null ribbon. Fixed-probe
aggregation applies the same bounded-memory policy.

## Geometry

The implementation computes:

- A: same question, N versus W;
- B: same question, W versus S;
- C: same question, N versus C;
- D: different questions, N versus N;
- E: different questions, W versus W;
- F: different questions, N versus W.

Unmatched groups use 100 deterministic derangements with no self-pairs. Reported
quantities include raw cosine, train-centered cosine, Euclidean distance divided
by training median residual norm, the identity/framing ratio, item-delta
alignment/energy, pairwise delta cosine, question-held-out four-way and N/W
framing classification, and paired cross-framing retrieval.

## Layer selection and compute

Layer selection uses validation only. For each layer:

1. compute the symmetric signed W pressure score;
2. subtract the mean of neutral P(c) damage, neutral accuracy damage, and
   neutral free-generation invalid-output rate;
3. require positive bidirectionality, neutral invalid-output and all-condition
   degeneration rates at most 1%, neutral accuracy damage at most 2 points,
   and learned effect above the 95th percentile control score;
4. evaluate the selected layer and available immediate neighbors on test.

If no layer is eligible, the highest selectivity layer may be carried forward
only as descriptive; the confirmatory direction-specificity hypothesis fails.

The planned test matrix has 4,737,600 strict-choice rows before fixed-probe
candidate scoring, which may add up to 23,688,000 candidate passes. The tiny run
must report measured throughput and projected accelerator-hours. No full
submission is permitted if safe batch size one makes the matrix infeasible.

`tests/test_controlled_activation_steering_real_model.py` is the opt-in GPU/BF16
gate. It is skipped locally. The tiny Slurm job supplies the approved manifest,
source run, controlled config, and report path, then binds the resulting
`real_model_bf16_gate.json` hash and exact CUDA/BF16 model, tokenizer, template,
source, manifest, and clean-Git identities into `compute_projection.json`.

## Commands

First inspection gate:

```bash
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
EXPERIMENT_RUN_ID=activation_steering_examples_qwen_csqa_20260725_v1 \
sbatch jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/inspect_examples_qwen_csqa.sbatch
```

Tiny dry run after eight-item approval:

```bash
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
QUESTION_MANIFEST=configs/experiments/activation_steering_preflight_8_20260725.jsonl \
EXPERIMENT_RUN_ID=activation_steering_qwen_csqa_tiny_20260725_v1 \
LAYERS=17,18 CONTROL_SEEDS=0,1 \
sbatch jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/tiny_dry_run_qwen_csqa.sbatch
```

Structural validation:

```bash
DRY_RUN=1 \
SYCOPHANCY_STORAGE_ROOT_OVERRIDE=/n/holystore01/LABS/barak_lab/Users/ishapira \
ACTIVATION_STEERING_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
EXPERIMENT_RUN_ID=activation_steering_controlled_preflight_20260725 \
bash jobs/sycophancy_bias_probe/activation_steering_controlled_sharded_20260725/submit_activation_steering_controlled_sharded_20260725.sh
```

The model-filtered full commands in the bundle README are **DO NOT RUN YET**.
Live submission additionally requires `ALLOW_FULL_SUBMISSION=1`, an existing
approved 1,000-row manifest, passing tests and numeric gates, plus explicit
`ACTIVATION_STEERING_INSPECTION_REPORT` and
`ACTIVATION_STEERING_TINY_COMPUTE_REPORT` paths. It also requires
`ACTIVATION_STEERING_FULL_GATE_APPROVAL`: a researcher-completed JSON record
whose exact Git commit and SHA-256 hashes bind the config, full question
manifest, Alpaca manifest, inspection report, and tiny compute report. The
validator rejects a dirty worktree or any false or missing review assertion.
The terminal aggregation stage also materializes fixed-probe, Alpaca utility,
and identity-versus-framing summaries under `supplementary_aggregate/`; these
stages are no longer dependency-only side artifacts.

A live full submission atomically reserves a never-before-used run root whose
identity hashes the controlled config, audited 1,000-question manifest, and
fixed Alpaca manifest. Existing roots and stale reservations are hard failures
and are never cleaned automatically.
