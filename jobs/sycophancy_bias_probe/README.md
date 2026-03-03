SLURM jobs for `run_sycophancy_bias_probe.py`:

- `fast_dirty.sbatch`: very quick sanity run.
- `medium.sbatch`: medium-scale run.
- `full.sbatch`: full dataset run.

All jobs:

- load `python/3.10.9-fasrc01`
- activate `conda` env `itai_ml_env`
- source `.env`
- enforce `HUGGINGFACE_HUB_CACHE` is set and not under `/home`
- set HF cache env vars so model/tokenizer/dataset cache uses lab storage

Submit examples:

```bash
sbatch jobs/sycophancy_bias_probe/fast_dirty.sbatch
sbatch jobs/sycophancy_bias_probe/medium.sbatch
sbatch jobs/sycophancy_bias_probe/full.sbatch
```

