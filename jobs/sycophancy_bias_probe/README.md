SLURM jobs for `run_sycophancy_bias_probe.py`:

- `fast_dirty.sbatch`: very quick sanity run.
- `fast_truthful_qa.sbatch`: very quick sanity run restricted to `truthful_qa`.
- `fast_dirty_aqua_mc.sbatch`: very quick AYS-derived MC sanity run restricted to `aqua_mc`.
- `medium.sbatch`: medium-scale run.
- `medium_aqua_mc.sbatch`: medium AYS-derived MC run restricted to `aqua_mc`.
- `full.sbatch`: full dataset run.
- `full_aqua_mc.sbatch`: full AYS-derived MC run restricted to `aqua_mc`.

All jobs:

- load `python/3.10.9-fasrc01`
- activate `conda` env `itai_ml_env`
- source `.env`
- enforce `HUGGINGFACE_HUB_CACHE` is set and not under `/home`
- set HF cache env vars so model/tokenizer/dataset cache uses lab storage

Submit examples:

```bash
sbatch jobs/sycophancy_bias_probe/fast_dirty.sbatch
sbatch jobs/sycophancy_bias_probe/fast_truthful_qa.sbatch
sbatch jobs/sycophancy_bias_probe/fast_dirty_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/medium.sbatch
sbatch jobs/sycophancy_bias_probe/medium_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/full.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc.sbatch
```
