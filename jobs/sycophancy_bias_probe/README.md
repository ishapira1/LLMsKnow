SLURM jobs for `run_sycophancy_bias_probe.py`:

Partition defaults in this directory:

- Local Hugging Face GPU jobs now request `gpu,seas_gpu,gpu_h200` so Slurm can pick the earliest-start stable full-GPU partition automatically.
- OpenAI sample-only jobs now request `shared,sapphire`; they do not allocate a GPU.
- `full_aqua_mc_n64.sbatch` remains on `gpu_test` as the explicit MIG/test queue script.
- Filenames ending in `_seas` are kept for continuity, even though the partition request is now broader than `seas_gpu`.

- `smoke_aqua_mc_auto.sh`: direct smoke/integrity run using `mistralai/Mistral-7B-Instruct-v0.2` on the AYS-derived `aqua_mc` slice, preferring GPU via `--device auto` and falling back to CPU when no accelerator is available, with an artifact-level integrity check after the pipeline finishes. The wrapper normalizes `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` / `HF_HOME` into a single Hugging Face cache path and passes it to the pipeline. The integrity CLI warns by default and only exits non-zero when passed `--strict`.
- `full_aqua_mc_gpt54nano_samples.sh`: direct run wrapper for the `aqua_mc` all-questions slice aligned to the `full_aqua_mc_mistral7b_auto_allq_l32_seas` setup, but using `gpt-5.4-nano` and `--sampling_only` so the saved sampling artifacts can be compared against the Mistral reference later. It sources `.env`, requires `OPENAI_API_KEY_FOR_PROJECT` (or `OPENAI_API_KEY`), defaults to bounded parallel OpenAI requests via `SAMPLE_BATCH_SIZE=8`, and runs integrity checks after sampling completes.
- `full_aqua_mc_gpt54nano_20260320_seas.sbatch`: dated CPU batch wrapper for `gpt-5.4-nano` on the `aqua_mc` all-questions slice. It reuses the sample-only OpenAI wrapper, so probes are skipped and the job / log names include `20260320`.
- `smoke_aqua_mc_cpu.sh`: compatibility alias for the historical job name; it forwards to `smoke_aqua_mc_auto.sh`.
- `aqua_mc_seas_common.sh`: shared helper used by the dated local-model GPU batch jobs. It keeps the smoke job's `aqua_mc` + `Mistral-7B-Instruct-v0.2` setup, normalizes the Hugging Face cache env vars, and omits strict-MC sampling flags that are normalized away anyway.
- `fast_aqua_mc_seas.sbatch`: `aqua_mc` fast preset using the stable full-GPU partition list, with 24 smoke questions and probe layers `1..8`.
- `medium_aqua_mc_seas.sbatch`: `aqua_mc` medium preset using the stable full-GPU partition list, with `max_questions=300` and probe layers `1..16`.
- `full_aqua_mc_seas.sbatch`: `aqua_mc` full preset using the stable full-GPU partition list, with all available questions and probe layers `1..32`.
- `full_aqua_mc_mistral7b_20260320_seas.sbatch`: dated local-GPU batch job for `mistralai/Mistral-7B-Instruct-v0.2`, matching the current full `aqua_mc` setup with probe layers `1..32`.
- `full_aqua_mc_llama31_8b_20260320_seas.sbatch`: dated local-GPU batch job for `meta-llama/Llama-3.1-8B-Instruct`, using the same full `aqua_mc` setup with probe layers `1..32`.
- `full_aqua_mc_qwen3_4b_20260320_seas.sbatch`: dated local-GPU batch job for `Qwen/Qwen3-4B-Instruct-2507`, using the same full `aqua_mc` setup with probe layers `1..32`.
- `submit_aqua_mc_20260320_seas.sh`: submits the dated `20260320` SEAS batch of four jobs: `gpt-5.4-nano` sample-only, `Mistral-7B-Instruct-v0.2`, `Llama-3.1-8B-Instruct`, and `Qwen3-4B-Instruct-2507`.
- `full_commonsense_qa_dataset5_gpt54nano_250q_20260320_seas.sbatch`: dated `dataset5` SEAS batch wrapper for `gpt-5.4-nano`, restricted to the `commonsense_qa` 250-question subset and pinned to `split_seed=5` so the same question pool is used across models. This remains sample-only.
- `full_commonsense_qa_dataset5_mistral7b_250q_20260320_seas.sbatch`: dated `dataset5` SEAS batch job for `mistralai/Mistral-7B-Instruct-v0.2`, with `dataset_name=commonsense_qa`, `ays_mc_datasets=commonsense_qa`, `max_questions=250`, `split_seed=5`, `seed=5`, and `probe_seed=5`.
- `full_commonsense_qa_dataset5_llama31_8b_250q_20260320_seas.sbatch`: dated `dataset5` SEAS batch job for `meta-llama/Llama-3.1-8B-Instruct`, with the same pinned `commonsense_qa` 250-question subset and seeds.
- `full_commonsense_qa_dataset5_qwen3_4b_250q_20260320_seas.sbatch`: dated `dataset5` SEAS batch job for `Qwen/Qwen3-4B-Instruct-2507`, with the same pinned `commonsense_qa` 250-question subset and seeds.
- `submit_commonsense_qa_dataset5_250q_20260320_seas.sh`: submits the dated `dataset5` 250-question SEAS batch of four `commonsense_qa` jobs.
- `full_commonsense_qa_llama31_8b_20260321_seas.sbatch`: dated SEAS batch job for `meta-llama/Llama-3.1-8B-Instruct` on the full `commonsense_qa` slice, using all available question groups, `split_seed=5`, `seed=5`, `probe_seed=5`, and the full model probe depth (the job passes a high layer ceiling and the pipeline clamps it to the model's actual number of layers). This is the overnight all-questions LLAMA job.
- `submit_commonsense_qa_full_llama31_8b_20260321_seas.sh`: submits the full-dataset LLAMA `commonsense_qa` overnight job.
- `full_commonsense_qa_qwen25_7b_20260322_seas.sbatch`: dated SEAS batch job for `Qwen/Qwen2.5-7B-Instruct` on the full `commonsense_qa` slice, intentionally matched to the latest full-dataset LLAMA job: same partition, GPU count, wall time, `mem=60G`, `split_seed=5`, `seed=5`, `probe_seed=5`, and full probe depth via `probe_layer_max=999`.
- `submit_commonsense_qa_full_qwen25_7b_20260322_seas.sh`: submits the full-dataset Qwen2.5 `commonsense_qa` overnight job.
- `full_arc_challenge_llama31_8b_20260321_seas.sbatch`: dated SEAS batch job for `meta-llama/Llama-3.1-8B-Instruct` on the full `arc_challenge` slice, using the ARC-native splits, `split_seed=5`, `seed=5`, `probe_seed=5`, and the full model probe depth (the job passes a high layer ceiling and the pipeline clamps it to the model's actual number of layers). This is the overnight all-questions LLAMA job for ARC-Challenge.
- `submit_arc_challenge_full_llama31_8b_20260321_seas.sh`: submits the full-dataset LLAMA `arc_challenge` overnight job.
- `full_arc_challenge_qwen25_7b_20260322_seas.sbatch`: dated SEAS batch job for `Qwen/Qwen2.5-7B-Instruct` on the full `arc_challenge` slice, intentionally matched to the latest full-dataset LLAMA ARC job: same partition, GPU count, wall time, `mem=20G`, `split_seed=5`, `seed=5`, `probe_seed=5`, and full probe depth via `probe_layer_max=999`.
- `submit_arc_challenge_full_qwen25_7b_20260322_seas.sh`: submits the full-dataset Qwen2.5 `arc_challenge` overnight job.
- `backfill_bias_probes_on_neutral_20260402_seas.sbatch`: array backfill job for the four main Llama/Qwen runs that rescored saved chosen bias probes on `neutral` prompts only.
- `submit_backfill_bias_probes_on_neutral_20260402_seas.sh`: submits the neutral-only chosen-probe backfill array job.
- `backfill_incorrect_suggestion_cross_family_20260407_seas.sbatch`: array backfill job that auto-discovers every saved run with `probe_bias_incorrect_suggestion` and rescored that chosen probe on the non-training prompt families by default: `neutral`, `doubt_correct`, `suggest_correct`, and `model_congruent_suggestion` when present. It excludes smoke runs by default, includes backup runs by default, and overwrites the canonical `probe_bias_incorrect_suggestion_all_templates` backfill unless `FORCE_BACKFILL=0`.
- `submit_backfill_incorrect_suggestion_cross_family_20260407_seas.sh`: computes the array size from the discovered runs and submits the incorrect-suggestion cross-family backfill job. Set `DRY_RUN=1` to print the discovered task list and the final `sbatch` command without submitting.
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
- set `#SBATCH --mail-type=END,FAIL` and `#SBATCH --mail-user=itaishapira@g.harvard.edu`

Submit examples:

```bash
bash jobs/sycophancy_bias_probe/smoke_aqua_mc_auto.sh
bash jobs/sycophancy_bias_probe/full_aqua_mc_gpt54nano_samples.sh
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_gpt54nano_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_mistral7b_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_llama31_8b_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_aqua_mc_qwen3_4b_20260320_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_aqua_mc_20260320_seas.sh
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_dataset5_gpt54nano_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_dataset5_mistral7b_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_dataset5_llama31_8b_250q_20260320_seas.sbatch
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_dataset5_qwen3_4b_250q_20260320_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_commonsense_qa_dataset5_250q_20260320_seas.sh
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_llama31_8b_20260321_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_commonsense_qa_full_llama31_8b_20260321_seas.sh
sbatch jobs/sycophancy_bias_probe/full_commonsense_qa_qwen25_7b_20260322_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_commonsense_qa_full_qwen25_7b_20260322_seas.sh
sbatch jobs/sycophancy_bias_probe/full_arc_challenge_llama31_8b_20260321_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_arc_challenge_full_llama31_8b_20260321_seas.sh
sbatch jobs/sycophancy_bias_probe/full_arc_challenge_qwen25_7b_20260322_seas.sbatch
bash jobs/sycophancy_bias_probe/submit_arc_challenge_full_qwen25_7b_20260322_seas.sh
bash jobs/sycophancy_bias_probe/submit_backfill_bias_probes_on_neutral_20260402_seas.sh
bash jobs/sycophancy_bias_probe/submit_backfill_incorrect_suggestion_cross_family_20260407_seas.sh
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
