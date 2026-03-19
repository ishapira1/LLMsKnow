# Probe Layout

This subpackage owns the probe pipeline and the probe-specific hidden-state feature extraction.

- `features.py`: assistant-token alignment plus single-layer and all-layer hidden-state features.
- `records.py`: lightweight record helpers such as completion-text lookup and deterministic subsampling.
- `select_layer.py`: validation-AUC layer selection across a layer grid.
- `train.py`: final probe fitting for one chosen layer.
- `score.py`: record-level probe scoring.
- `__init__.py`: stable public exports for `llmssycoph.probes`.
