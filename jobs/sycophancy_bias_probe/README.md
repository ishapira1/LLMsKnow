SLURM jobs for `run_sycophancy_bias_probe.py`:

- `smoke_aqua_mc_auto.sh`: direct smoke/integrity run using `mistralai/Mistral-7B-Instruct-v0.2` on the AYS-derived `aqua_mc` slice, preferring GPU via `--device auto` and falling back to CPU when no accelerator is available, with an artifact-level integrity check after the pipeline finishes. The wrapper normalizes `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` / `HF_HOME` into a single Hugging Face cache path and passes it to the pipeline. The integrity CLI warns by default and only exits non-zero when passed `--strict`.
- `smoke_aqua_mc_cpu.sh`: compatibility alias for the historical job name; it forwards to `smoke_aqua_mc_auto.sh`.
- `aqua_mc_seas_common.sh`: shared helper used by the `seas_gpu` batch jobs. It keeps the smoke job's `aqua_mc` + `Mistral-7B-Instruct-v0.2` setup, normalizes the Hugging Face cache env vars, and omits strict-MC sampling flags that are normalized away anyway.
- `fast_aqua_mc_seas.sbatch`: `aqua_mc` fast preset for `seas_gpu` with 24 smoke questions and probe layers `1..8`.
- `medium_aqua_mc_seas.sbatch`: `aqua_mc` medium preset for `seas_gpu` with `max_questions=300` and probe layers `1..16`.
- `full_aqua_mc_seas.sbatch`: `aqua_mc` full preset for `seas_gpu` with all available questions and probe layers `1..32`.
- `fast_dirty.sbatch`: very quick sanity run.
- `fast_truthful_qa.sbatch`: very quick sanity run restricted to `truthful_qa`.
- `fast_aqua_mc.sbatch`: very quick AYS-derived MC sanity run restricted to `aqua_mc`.
- `fast_dirty_aqua_mc.sbatch`: legacy AQuA smoke job name; prefer `fast_aqua_mc.sbatch`.
- `medium.sbatch`: medium-scale run.
- `medium_aqua_mc.sbatch`: medium AYS-derived MC run restricted to `aqua_mc`.
- `full.sbatch`: full dataset run.
- `full_aqua_mc.sbatch`: full AYS-derived MC run restricted to `aqua_mc`.
- `submit_aqua_mc.sh`: submit the final overnight AQuA-only trio (`fast_aqua_mc`, `medium_aqua_mc`, `full_aqua_mc`).

All jobs:

- load `python/3.10.9-fasrc01`
- activate `conda` env `itai_ml_env`
- source `.env`
- prepend `$REPO_DIR/src` to `PYTHONPATH` so `llmssycoph` resolves without an install step
- enforce `HUGGINGFACE_HUB_CACHE` is set and not under `/home`
- set HF cache env vars so model/tokenizer/dataset cache uses lab storage

Submit examples:

```bash
bash jobs/sycophancy_bias_probe/smoke_aqua_mc_auto.sh
sbatch jobs/sycophancy_bias_probe/fast_aqua_mc_seas.sbatch
sbatch jobs/sycophancy_bias_probe/medium_aqua_mc_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_seas.sbatch
sbatch jobs/sycophancy_bias_probe/fast_dirty.sbatch
sbatch jobs/sycophancy_bias_probe/fast_truthful_qa.sbatch
sbatch jobs/sycophancy_bias_probe/fast_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/medium.sbatch
sbatch jobs/sycophancy_bias_probe/medium_aqua_mc.sbatch
sbatch jobs/sycophancy_bias_probe/full.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc.sbatch
bash jobs/sycophancy_bias_probe/submit_aqua_mc.sh
```
