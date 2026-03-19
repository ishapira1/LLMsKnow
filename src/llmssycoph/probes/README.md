# Probe Layout

This subpackage owns the probe pipeline and the probe-specific hidden-state feature extraction.

- Probe-example assembly lives just outside this package in `llmssycoph.grading.probe_data`.
- For generation-based paths, probe examples are the sampled completions themselves.
- For `mc_mode=strict_mc`, probe examples are teacher-forced `(prompt, answer_choice)` candidate rows, weighted by model choice probability by default.
- The selected-answer probe score is written back to `sampling/sampled_responses.csv`, while the full per-choice table is saved as `probes/probe_candidate_scores.csv`.

- `features.py`: assistant-token alignment plus single-layer and all-layer hidden-state features.
- `records.py`: lightweight record helpers such as completion-text lookup and deterministic subsampling.
- `select_layer.py`: validation-AUC layer selection across a layer grid.
- `train.py`: final probe fitting for one chosen layer.
- `score.py`: record-level probe scoring.
- `metrics.py`: shared probe metric definitions plus split-wise evaluation/provenance summaries.
- `artifacts.py`: writes `probes/all_probes/` and `probes/chosen_probe/` artifact bundles.
- `__init__.py`: stable public exports for `llmssycoph.probes`.
