# Claim-3 Presentation Viewer

This folder contains a small static local website for presenting the packaged main-run claim-3 matrix.

## Refresh the data bundle

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/export_claim3_presentation_bundle.py
```

That command reads:

`results/sycophancy_bias_probe/analysis_exports/claim3_model_probe_train_eval_breakdown_main_runs/claim3_model_probe_train_eval_breakdown_main_runs_long.csv`

and writes:

`viewer/claim3_presentation/data/claim3_presentation_bundle_main_runs.json`

## Serve locally

```bash
python -m http.server 8000 --directory viewer/claim3_presentation
```

Then open:

`http://localhost:8000`

## Notes

- Scope is the current packaged main runs only.
- `All` selectors use the precomputed equal-weight or prompt-weighted views in the bundle.
- Probe metrics use the saved chosen best-layer probe for each train-on family.
