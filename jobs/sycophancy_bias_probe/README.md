SLURM jobs for `run_sycophancy_bias_probe.py`:

- `smoke_aqua_mc_auto.sh`: direct smoke/integrity run using `mistralai/Mistral-7B-Instruct-v0.2` on the AYS-derived `aqua_mc` slice, preferring GPU via `--device auto` and falling back to CPU when no accelerator is available, with an artifact-level integrity check after the pipeline finishes. The wrapper normalizes `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` / `HF_HOME` into a single Hugging Face cache path and passes it to the pipeline. The integrity CLI warns by default and only exits non-zero when passed `--strict`.
- `full_aqua_mc_gpt54nano_samples.sh`: direct run wrapper for the `aqua_mc` all-questions slice aligned to the `full_aqua_mc_mistral7b_auto_allq_l32_seas` setup, but using `gpt-5.4-nano` and `--sampling_only` so the saved sampling artifacts can be compared against the Mistral reference later. It sources `.env`, requires `OPENAI_API_KEY_FOR_PROJECT` (or `OPENAI_API_KEY`), defaults to bounded parallel OpenAI requests via `SAMPLE_BATCH_SIZE=8`, and runs integrity checks after sampling completes.
- `full_aqua_mc_gpt54mini_20260320_seas.sbatch`: dated `seas_gpu` batch wrapper for `gpt-5.4-mini` on the `aqua_mc` all-questions slice. It reuses the sample-only OpenAI wrapper, so probes are skipped and the job / log names include `20260320`.
- `smoke_aqua_mc_cpu.sh`: compatibility alias for the historical job name; it forwards to `smoke_aqua_mc_auto.sh`.
- `aqua_mc_seas_common.sh`: shared helper used by the `seas_gpu` batch jobs. It keeps the smoke job's `aqua_mc` + `Mistral-7B-Instruct-v0.2` setup, normalizes the Hugging Face cache env vars, and omits strict-MC sampling flags that are normalized away anyway.
- `fast_aqua_mc_seas.sbatch`: `aqua_mc` fast preset for `seas_gpu` with 24 smoke questions and probe layers `1..8`.
- `medium_aqua_mc_seas.sbatch`: `aqua_mc` medium preset for `seas_gpu` with `max_questions=300` and probe layers `1..16`.
- `full_aqua_mc_seas.sbatch`: `aqua_mc` full preset for `seas_gpu` with all available questions and probe layers `1..32`.
- `full_aqua_mc_mistral7b_20260320_seas.sbatch`: dated `seas_gpu` batch job for `mistralai/Mistral-7B-Instruct-v0.2`, matching the current full `aqua_mc` SEAS setup with probe layers `1..32`.
- `full_aqua_mc_llama31_8b_20260320_seas.sbatch`: dated `seas_gpu` batch job for `meta-llama/Llama-3.1-8B-Instruct`, using the same full `aqua_mc` SEAS setup with probe layers `1..32`.
- `full_aqua_mc_qwen3_4b_20260320_seas.sbatch`: dated `seas_gpu` batch job for `Qwen/Qwen3-4B-Instruct-2507`, using the same full `aqua_mc` SEAS setup with probe layers `1..32`.
- `submit_aqua_mc_20260320_seas.sh`: submits the dated `20260320` SEAS batch of four jobs: `gpt-5.4-mini` sample-only, `Mistral-7B-Instruct-v0.2`, `Llama-3.1-8B-Instruct`, and `Qwen3-4B-Instruct-2507`.
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
bash jobs/sycophancy_bias_probe/full_aqua_mc_gpt54nano_samples.sh
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_gpt54mini_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_mistral7b_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_llama31_8b_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_qwen3_4b_20260320_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_aqua_mc_20260320_seas.sh
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
